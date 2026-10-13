"""Model clients: an OpenAI-compatible HTTP client, a local Hugging Face client, and an offline fake.

The HTTP client uses only the standard library and works with any provider
exposing an OpenAI-style ``POST {base_url}/chat/completions`` endpoint. The
API key is read from config or the MAS_API_KEY environment variable and is
never written to logs.

The Hugging Face client loads a model with ``transformers`` and generates in
this process, on CPU or, with ``--gpu``, on CUDA GPUs. ``torch`` and
``transformers`` are imported only when that backend is chosen, so the rest of
the project stays standard-library only.

The fake client returns random but schema-valid JSON. It exists only to test
the pipeline without keys; its outputs carry no scientific meaning.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
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


DEFAULT_HF_MODEL = "Qwen/Qwen3.8-27B"
BACKENDS = ("api", "hf", "fake")


def strip_thinking(text: str, started_in_prompt: bool = False) -> str:
    """Return only the final answer of a reply that may contain a reasoning block.

    Qwen chat templates open ``<think>`` inside the prompt when thinking is on, so the
    generated text then holds only the closing tag. A block that never closes means the
    token budget ran out mid-reasoning: there is no answer, so return an empty string.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if started_in_prompt or "<think>" in text:
        return ""
    return text.strip()


class HFLocalClient:
    """Chat model loaded with Hugging Face ``transformers`` and run in this process.

    Settings come from ``llm.hf`` (model, dtype, enable_thinking, top_p, top_k,
    repetition_penalty, trust_remote_code, max_new_tokens) plus the shared
    ``llm.temperature`` and ``llm.max_tokens``. A temperature of ``null`` or 0 means
    greedy decoding. With ``gpu`` the weights are placed with ``device_map="auto"``,
    which spreads a model over every visible CUDA GPU; without it everything runs on CPU.
    """

    serial = True  # one model instance, so calls are made one at a time

    def __init__(self, settings: Dict, gpu: bool, seed: int) -> None:
        hf = dict(settings.get("hf", {}))
        self.model_name = hf.get("model") or DEFAULT_HF_MODEL
        self.enable_thinking = bool(hf.get("enable_thinking", False))
        trust = bool(hf.get("trust_remote_code", False))
        self.temperature = settings.get("temperature")
        self.max_new_tokens = int(hf.get("max_new_tokens") or settings.get("max_tokens") or 800)
        self.top_p = hf.get("top_p")
        self.top_k = hf.get("top_k")
        self.repetition_penalty = float(hf.get("repetition_penalty", 1.0))
        self.lock = threading.Lock()
        try:
            import torch
            import transformers
        except ImportError as error:
            raise SystemExit(
                "The Hugging Face backend needs extra packages. Install them with:\n"
                "    pip install -r requirements.txt"
            ) from error
        if gpu and not torch.cuda.is_available():
            raise SystemExit(
                "--gpu was given but PyTorch cannot see a CUDA GPU. Run without --gpu to use the "
                "CPU, or check the CUDA installation (python -c \"import torch; "
                "print(torch.cuda.is_available())\")."
            )
        self._torch = torch
        torch.manual_seed(seed)
        print(f"Loading {self.model_name} on {'GPU' if gpu else 'CPU'}; the first load downloads "
              "the weights and can take a long time.", flush=True)
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=trust)
            config = transformers.AutoConfig.from_pretrained(self.model_name, trust_remote_code=trust)
            architecture = (getattr(config, "architectures", None) or [None])[0]
            model_class = getattr(transformers, architecture, None) if architecture else None
            load_args = {"dtype": hf.get("dtype", "auto"), "trust_remote_code": trust}
            if gpu:
                load_args["device_map"] = "auto"
            self.model = (model_class or transformers.AutoModelForCausalLM).from_pretrained(
                self.model_name, **load_args)
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(
                f"Could not load {self.model_name}: {error}\nCheck the model id and network access, "
                "and that transformers>=5.8.0 is installed (see requirements.txt)."
            ) from error
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def name(self) -> str:
        return f"hf:{self.model_name}"

    def complete(self, messages: List[Dict[str, str]], spec: Dict) -> Completion:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        sample = self.temperature is not None and self.temperature > 0
        generation = {"max_new_tokens": self.max_new_tokens, "do_sample": sample,
                      "pad_token_id": self.tokenizer.pad_token_id}
        if sample:
            generation["temperature"] = float(self.temperature)
            if self.top_p is not None:
                generation["top_p"] = float(self.top_p)
            if self.top_k is not None:
                generation["top_k"] = int(self.top_k)
        if self.repetition_penalty != 1.0:
            generation["repetition_penalty"] = self.repetition_penalty
        started = time.monotonic()
        try:
            with self.lock, self._torch.inference_mode():
                output = self.model.generate(**inputs, **generation)
        except RuntimeError as error:  # includes CUDA out-of-memory
            raise ModelError(f"generation failed: {error}") from error
        prompt_tokens = int(inputs["input_ids"].shape[1])
        new_tokens = output[0][prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return Completion(
            strip_thinking(text, started_in_prompt=self.enable_thinking),
            {"prompt_tokens": prompt_tokens, "completion_tokens": int(new_tokens.shape[0])},
            time.monotonic() - started,
        )


def resolve_backend(config_backend: Optional[str], fake: bool = False, backend: Optional[str] = None,
                    gpu: bool = False, hf_model: Optional[str] = None) -> str:
    """Pick the backend: explicit choice, else implied by --gpu / --hf-model, else config, else api."""
    chosen = [name for name in ("fake" if fake else None, backend) if name]
    if len(set(chosen)) > 1:
        raise ValueError("choose only one of --fake, --api and --hf")
    explicit = chosen[0] if chosen else None
    if gpu or hf_model:
        if explicit in ("api", "fake"):
            raise ValueError("--gpu and --hf-model apply only to the Hugging Face backend, "
                             "not to --api or --fake")
        explicit = "hf"
    resolved = explicit or config_backend or "api"
    if resolved not in BACKENDS:
        raise ValueError(f"unknown backend {resolved!r}; use one of {', '.join(BACKENDS)}")
    return resolved


def build_client(settings: Dict, backend: str, seed: int, gpu: bool = False,
                 hf_model: Optional[str] = None):
    if backend == "fake":
        return FakeClient(seed=seed, invalid_rate=float(settings.get("fake_invalid_rate", 0.0)))
    if backend == "hf":
        hf = dict(settings.get("hf", {}))
        if hf_model:
            hf["model"] = hf_model
        return HFLocalClient({**settings, "hf": hf}, gpu=gpu, seed=seed)
    missing = [key for key in ("base_url", "api_key", "model") if not settings.get(key)]
    if missing:
        raise SystemExit(
            "Missing LLM settings: " + ", ".join(missing) + ". Fill them in config.json "
            "(or set MAS_BASE_URL / MAS_API_KEY / MAS_MODEL), or run with --fake for an offline "
            "test or --hf for a local Hugging Face model."
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
