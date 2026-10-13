"""Model clients: an OpenAI-compatible HTTP client, a local Hugging Face client, a vLLM client,
and an offline fake.

The HTTP client uses only the standard library and works with any provider
exposing an OpenAI-style ``POST {base_url}/chat/completions`` endpoint. The
API key is read from config or the MAS_API_KEY environment variable and is
never written to logs.

The Hugging Face client loads a model with ``transformers`` and generates in
this process, on CPU or, with ``--gpu``, on CUDA GPUs. ``torch`` and
``transformers`` are imported only when that backend is chosen, so the rest of
the project stays standard-library only.

The vLLM client talks to a vLLM server over its OpenAI-compatible API, so it needs no
extra Python packages here; the ``vllm`` command only has to be installed. It either
starts ``vllm serve`` itself on a free local port and stops it afterwards, or connects
to a server that is already running. Because vLLM batches concurrent requests, trials
run in parallel on this backend.

The fake client returns random but schema-valid JSON. It exists only to test
the pipeline without keys; its outputs carry no scientific meaning.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
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
DEFAULT_VLLM_MODEL = DEFAULT_HF_MODEL
BACKENDS = ("api", "hf", "vllm", "fake")


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


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class VLLMClient(OpenAICompatibleClient):
    """Chat client for a vLLM server that this process starts, or that is already running.

    Without ``llm.vllm.base_url`` it launches ``vllm serve`` on a free local port, waits until
    ``GET /health`` succeeds, and stops the server in ``close()`` and at interpreter exit. With
    ``base_url`` it only connects to a server you started yourself. Requests go to vLLM's
    OpenAI-compatible ``/v1/chat/completions``, so trials can run in parallel and vLLM batches
    them. A temperature of ``null`` or 0 means greedy decoding, as for the Hugging Face backend.
    """

    def __init__(self, settings: Dict, seed: int, log_dir: Optional[str] = None) -> None:
        vllm = dict(settings.get("vllm", {}))
        self.model_name = vllm.get("model") or DEFAULT_VLLM_MODEL
        self.enable_thinking = bool(vllm.get("enable_thinking", False))
        self._stop_timeout = float(vllm.get("stop_timeout_seconds", 30))
        self._sigterm = None
        self._process: Optional[subprocess.Popen] = None
        self._log_handle = None
        self._log_path = os.path.join(log_dir or tempfile.gettempdir(), "vllm_server.log")
        external = str(vllm.get("base_url") or "").strip().rstrip("/")
        if external.endswith("/v1"):
            external = external[: -len("/v1")]
        self.owns_server = not external
        port = int(vllm.get("port") or 0) or (_free_port() if self.owns_server else 0)
        self.server_url = external or f"http://127.0.0.1:{port}"
        extra_body: Dict = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        for key in ("top_p", "top_k", "repetition_penalty"):
            value = vllm.get(key)
            if value is not None and not (key == "repetition_penalty" and float(value) == 1.0):
                extra_body[key] = value
        extra_body.update(settings.get("extra_body") or {})
        temperature = settings.get("temperature")
        super().__init__({
            "base_url": f"{self.server_url}/v1",
            "api_key": vllm.get("api_key") or "EMPTY",
            "model": self.model_name,
            "temperature": 0 if temperature is None else temperature,
            "max_tokens": settings.get("max_tokens"),
            "max_tokens_field": settings.get("max_tokens_field", "max_tokens"),
            "json_mode": settings.get("json_mode", False),
            "timeout_seconds": settings.get("timeout_seconds", 120),
            "max_http_retries": settings.get("max_http_retries", 3),
            "extra_body": extra_body,
            "extra_headers": settings.get("extra_headers") or {},
        })
        if self.owns_server:
            self._launch(vllm, seed, port)
            self._wait_until_ready(float(vllm.get("startup_timeout_seconds", 1800)),
                                   float(vllm.get("startup_poll_seconds", 2)))
        elif not self._healthy():
            raise SystemExit(
                f"Cannot reach a vLLM server at {self.server_url} (GET /health failed). Start it "
                "first, or leave llm.vllm.base_url / --vllm-url unset so this program starts one."
            )

    @property
    def name(self) -> str:
        return f"vllm:{self.model_name}"

    def _launch(self, vllm: Dict, seed: int, port: int) -> None:
        command = [str(part) for part in (vllm.get("command") or ["vllm"])]
        sibling = os.path.join(os.path.dirname(sys.executable), "vllm")
        if command == ["vllm"] and os.path.exists(sibling):
            command = [sibling]
        if shutil.which(command[0]) is None:
            raise SystemExit(
                "The vLLM backend needs the `vllm` command. Install vLLM with `pip install vllm` "
                "(Linux with NVIDIA GPUs, see https://docs.vllm.ai/), or start a server yourself "
                "and pass --vllm-url."
            )
        tensor_parallel = vllm.get("tensor_parallel_size")
        if tensor_parallel is not None and int(tensor_parallel) < 1:
            raise SystemExit("tensor_parallel_size must be at least 1")
        args = command + ["serve", self.model_name, "--host", "127.0.0.1", "--port", str(port),
                          "--dtype", str(vllm.get("dtype", "auto")), "--seed", str(seed)]
        if tensor_parallel is not None:
            args += ["--tensor-parallel-size", str(int(tensor_parallel))]
        if vllm.get("max_model_len"):
            args += ["--max-model-len", str(vllm["max_model_len"])]
        if vllm.get("gpu_memory_utilization") is not None:
            args += ["--gpu-memory-utilization", str(vllm["gpu_memory_utilization"])]
        if self.enable_thinking and vllm.get("reasoning_parser"):
            args += ["--reasoning-parser", str(vllm["reasoning_parser"])]
        if vllm.get("trust_remote_code"):
            args.append("--trust-remote-code")
        args += [str(extra) for extra in vllm.get("extra_args") or []]
        os.makedirs(os.path.dirname(os.path.abspath(self._log_path)), exist_ok=True)
        print(f"Starting vLLM for {self.model_name} at {self.server_url}; server log: "
              f"{self._log_path}", flush=True)
        try:
            self._log_handle = open(self._log_path, "ab")
            self._process = subprocess.Popen(
                args, stdin=subprocess.DEVNULL, stdout=self._log_handle, stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix")
        except OSError as error:
            raise SystemExit(f"Could not start vLLM ({args[0]}): {error}") from error
        atexit.register(self.close)
        self._install_sigterm_handler()

    def _install_sigterm_handler(self) -> None:
        """Turn SIGTERM into a normal exit so the server is stopped, unless the caller set one."""
        if threading.current_thread() is not threading.main_thread():
            return
        previous = signal.getsignal(signal.SIGTERM)
        if previous != signal.SIG_DFL:
            return

        def handler(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, handler)
        self._sigterm = (handler, previous)

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.server_url}/health", timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _log_tail(self, lines: int = 25) -> str:
        try:
            with open(self._log_path, "rb") as handle:
                return b"".join(handle.readlines()[-lines:]).decode("utf-8", errors="replace")
        except OSError:
            return "(no log available)"

    def _wait_until_ready(self, timeout: float, poll: float) -> None:
        deadline = time.monotonic() + timeout
        ready = False
        try:
            while True:
                code = self._process.poll()
                if code is not None:
                    raise SystemExit(f"vLLM exited with code {code} before it was ready. Last lines "
                                     f"of {self._log_path}:\n{self._log_tail()}")
                if self._healthy():
                    ready = True
                    return
                if time.monotonic() >= deadline:
                    raise SystemExit(
                        f"vLLM was not ready after {timeout:.0f} s. Large models take a long time "
                        f"to load; raise llm.vllm.startup_timeout_seconds. Log: {self._log_path}")
                time.sleep(poll)
        finally:
            if not ready:
                self.close()

    def close(self) -> None:
        """Stop the server this client started. Safe to call more than once."""
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=self._stop_timeout)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self._sigterm is not None:
            handler, previous = self._sigterm
            self._sigterm = None
            if (threading.current_thread() is threading.main_thread()
                    and signal.getsignal(signal.SIGTERM) is handler):
                signal.signal(signal.SIGTERM, previous)

    def complete(self, messages: List[Dict[str, str]], spec: Dict) -> Completion:
        completion = super().complete(messages, spec)
        if "</think>" in completion.text:  # a server without a reasoning parser leaves it in
            completion.text = completion.text.rsplit("</think>", 1)[1].strip()
        return completion


def resolve_backend(config_backend: Optional[str], fake: bool = False, backend: Optional[str] = None,
                    gpu: bool = False, hf_model: Optional[str] = None,
                    vllm_model: Optional[str] = None, vllm_url: Optional[str] = None,
                    tensor_parallel_size: Optional[int] = None) -> str:
    """Pick the backend: explicit choice, else implied by a backend-specific flag, else config."""
    chosen = [name for name in ("fake" if fake else None, backend) if name]
    if len(set(chosen)) > 1:
        raise ValueError("choose only one of --fake, --api, --hf and --vllm")
    explicit = chosen[0] if chosen else None
    hf_flags = bool(gpu or hf_model)
    vllm_flags = bool(vllm_model or vllm_url or tensor_parallel_size)
    if hf_flags and vllm_flags:
        raise ValueError("--gpu and --hf-model belong to --hf; --vllm-model, --vllm-url and "
                         "--tensor-parallel-size belong to --vllm. Do not mix them")
    if hf_flags:
        if explicit in ("api", "fake", "vllm"):
            raise ValueError("--gpu and --hf-model apply only to the Hugging Face backend, "
                             "not to --api, --vllm or --fake")
        explicit = "hf"
    if vllm_flags:
        if explicit in ("api", "fake", "hf"):
            raise ValueError("--vllm-model, --vllm-url and --tensor-parallel-size apply only to "
                             "the vLLM backend, not to --api, --hf or --fake")
        explicit = "vllm"
    resolved = explicit or config_backend or "api"
    if resolved not in BACKENDS:
        raise ValueError(f"unknown backend {resolved!r}; use one of {', '.join(BACKENDS)}")
    return resolved


def build_client(settings: Dict, backend: str, seed: int, gpu: bool = False,
                 hf_model: Optional[str] = None, vllm_model: Optional[str] = None,
                 vllm_url: Optional[str] = None, tensor_parallel_size: Optional[int] = None,
                 log_dir: Optional[str] = None):
    if backend == "fake":
        return FakeClient(seed=seed, invalid_rate=float(settings.get("fake_invalid_rate", 0.0)))
    if backend == "vllm":
        vllm = dict(settings.get("vllm", {}))
        if vllm_model:
            vllm["model"] = vllm_model
        if vllm_url:
            vllm["base_url"] = vllm_url
        if tensor_parallel_size is not None:
            vllm["tensor_parallel_size"] = tensor_parallel_size
        return VLLMClient({**settings, "vllm": vllm}, seed=seed, log_dir=log_dir)
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
            "test, --hf for a local Hugging Face model or --vllm for a vLLM server."
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
