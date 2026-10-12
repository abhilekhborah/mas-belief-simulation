"""Model clients: an OpenAI-compatible HTTP client and an offline fake.

The HTTP client uses only the standard library and works with any provider
exposing an OpenAI-style ``POST {base_url}/chat/completions`` endpoint. The
API key is read from config or the MAS_API_KEY environment variable and is
never written to logs.

The fake client returns random but schema-valid JSON. It exists only to test
the pipeline without keys; its outputs carry no scientific meaning.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .prompts import GROUP_OUTCOMES
from .scenarios import LETTERS


@dataclass
class Completion:
    text: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_seconds: float = 0.0


class ModelError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, settings: Dict) -> None:
        self.base_url = settings["base_url"].rstrip("/")
        self.api_key = settings["api_key"]
        self.model = settings["model"]
        self.temperature = settings.get("temperature")
        self.max_tokens = settings.get("max_tokens")
        self.max_tokens_field = settings.get("max_tokens_field", "max_tokens")
        self.json_mode = bool(settings.get("json_mode", False))
        self.timeout = float(settings.get("timeout_seconds", 120))
        self.max_http_retries = int(settings.get("max_http_retries", 3))
        self.extra_body = dict(settings.get("extra_body", {}))
        self.extra_headers = dict(settings.get("extra_headers", {}))

    @property
    def name(self) -> str:
        return self.model

    def complete(self, messages: List[Dict[str, str]], spec: Dict) -> Completion:
        body: Dict = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body[self.max_tokens_field] = self.max_tokens
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        headers.update(self.extra_headers)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error = "unknown error"
        for attempt in range(self.max_http_retries + 1):
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = payload["choices"][0]["message"]["content"] or ""
                usage = {key: value for key, value in (payload.get("usage") or {}).items()
                         if isinstance(value, int)}
                return Completion(text, usage, time.monotonic() - started)
            except urllib.error.HTTPError as error:
                last_error = f"HTTP {error.code}"
                if error.code not in (408, 409, 429) and error.code < 500:
                    raise ModelError(f"{last_error}: {error.read()[:300]!r}") from error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last_error = f"network error: {error}"
            except (KeyError, IndexError, ValueError) as error:
                raise ModelError(f"unexpected response format: {error}") from error
            if attempt < self.max_http_retries:
                time.sleep(min(2 ** attempt, 30))
        raise ModelError(f"request failed after retries: {last_error}")


def _random_distribution(rng: random.Random, keys: Sequence[str]) -> Dict[str, float]:
    cuts = sorted(rng.randint(0, 100) for _ in range(len(keys) - 1))
    parts = [high - low for low, high in zip([0] + cuts, cuts + [100])]
    return {key: part / 100 for key, part in zip(keys, parts)}


class FakeClient:
    """Offline client producing schema-valid random JSON for pipeline tests."""

    def __init__(self, seed: int = 0, invalid_rate: float = 0.0) -> None:
        self.seed = seed
        self.invalid_rate = invalid_rate

    @property
    def name(self) -> str:
        return "fake-offline"

    def complete(self, messages: List[Dict[str, str]], spec: Dict) -> Completion:
        digest = hashlib.sha256(
            (str(self.seed) + json.dumps(messages, sort_keys=True)).encode("utf-8")
        ).hexdigest()
        rng = random.Random(digest)
        if rng.random() < self.invalid_rate:
            return Completion("not json")
        kind = spec["call_kind"]
        if kind == "private_probe":
            payload: Dict = {"private_belief": _random_distribution(rng, LETTERS)}
            if spec.get("peers"):
                payload["peer_forecasts"] = {
                    peer: _random_distribution(rng, LETTERS) for peer in spec["peers"]
                }
            if spec.get("ask_group"):
                payload["group_forecast"] = _random_distribution(rng, GROUP_OUTCOMES)
        elif kind == "public_turn":
            payload = {"public_answer": rng.choice(LETTERS), "public_message": "I have a view."}
        else:
            payload = {"vote": rng.choice(LETTERS), "rationale": "Based on the facts."}
        return Completion(json.dumps(payload))


def build_client(settings: Dict, fake: bool, seed: int):
    if fake:
        return FakeClient(seed=seed, invalid_rate=float(settings.get("fake_invalid_rate", 0.0)))
    missing = [key for key in ("base_url", "api_key", "model") if not settings.get(key)]
    if missing:
        raise SystemExit(
            "Missing LLM settings: " + ", ".join(missing) + ". Fill them in config.json "
            "(or set MAS_BASE_URL / MAS_API_KEY / MAS_MODEL), or run with --fake for an offline test."
        )
    return OpenAICompatibleClient(settings)


def parse_json_object(text: str) -> Optional[Dict]:
    """Extract the first JSON object from a reply, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
