"""Tests for the Hugging Face backend that run without torch or transformers installed.

The real libraries are replaced by small stubs, so these tests check our own logic: backend
selection, chat-template arguments, generation settings, reply decoding and thinking-block
removal, GPU handling, and that the model is called one request at a time.
"""

import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mas_sim.llm import (DEFAULT_HF_MODEL, HFLocalClient, ModelError, build_client,  # noqa: E402
                         resolve_backend, strip_thinking)


class StubTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        return StubVector(self.rows[index])


class StubVector:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def __getitem__(self, item):
        return StubVector(self.values[item]) if isinstance(item, slice) else self.values[item]


class StubInputs(dict):
    def to(self, device):
        self.device = device
        return self


def build_stubs(cuda_available=False, reply_tokens=(7, 8, 9), generate_error=None):
    log = {"template_calls": [], "generate_calls": [], "load_calls": [], "devices": []}

    class Tokenizer:
        pad_token_id = None
        pad_token = None
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            log["load_calls"].append(("tokenizer", name, kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            log["template_calls"].append(kwargs)
            return "PROMPT"

        def __call__(self, text, **kwargs):
            return StubInputs(input_ids=StubTensor([[1, 2, 3, 4]]))

        def decode(self, tokens, skip_special_tokens=True):
            return "".join(chr(ord("a") + value) for value in tokens.values)

    class Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            log["load_calls"].append(("model", name, kwargs))
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            log["generate_calls"].append(kwargs)
            if generate_error:
                raise generate_error
            return StubTensor([[1, 2, 3, 4, *reply_tokens]])

    class Config:
        architectures = ["StubArchitecture"]

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

    transformers = types.SimpleNamespace(
        AutoTokenizer=Tokenizer, AutoConfig=Config, AutoModelForCausalLM=Model,
        StubArchitecture=Model)

    class InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        manual_seed=lambda seed: log.setdefault("seed", seed),
        inference_mode=InferenceMode)
    return {"torch": torch, "transformers": transformers}, log


def make_client(settings=None, gpu=False, **stub_options):
    modules, log = build_stubs(**stub_options)
    base = {"temperature": 0.7, "max_tokens": 50, "hf": {"model": "stub/model", "top_p": 0.8, "top_k": 20}}
    if settings:
        base["hf"].update(settings.pop("hf", {}))
        base.update(settings)
    with mock.patch.dict(sys.modules, modules):
        client = HFLocalClient(base, gpu=gpu, seed=5)
    return client, log


class BackendSelectionTests(unittest.TestCase):
    def test_default_is_api_and_config_can_choose(self):
        self.assertEqual(resolve_backend(None), "api")
        self.assertEqual(resolve_backend("hf"), "hf")

    def test_flags_override_config(self):
        self.assertEqual(resolve_backend("api", fake=True), "fake")
        self.assertEqual(resolve_backend("hf", backend="api"), "api")

    def test_gpu_and_model_name_imply_hf(self):
        self.assertEqual(resolve_backend("api", gpu=True), "hf")
        self.assertEqual(resolve_backend("api", hf_model="x/y"), "hf")

    def test_conflicts_are_rejected(self):
        for kwargs in ({"fake": True, "gpu": True}, {"backend": "api", "gpu": True},
                       {"backend": "api", "hf_model": "x/y"}, {"fake": True, "backend": "hf"}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                resolve_backend(None, **kwargs)
        with self.assertRaises(ValueError):
            resolve_backend("nonsense")

    def test_default_model_is_qwen_3_8_27b(self):
        self.assertEqual(DEFAULT_HF_MODEL, "Qwen/Qwen3.8-27B")


class ThinkingTests(unittest.TestCase):
    def test_plain_reply_is_unchanged(self):
        self.assertEqual(strip_thinking('  {"a": 1} '), '{"a": 1}')

    def test_reasoning_block_is_removed(self):
        self.assertEqual(strip_thinking("<think>hmm</think>\n\n{\"a\": 1}"), '{"a": 1}')
        self.assertEqual(strip_thinking("reasoning only</think>\n\nanswer", started_in_prompt=True), "answer")

    def test_unfinished_reasoning_has_no_answer(self):
        self.assertEqual(strip_thinking("still thinking", started_in_prompt=True), "")
        self.assertEqual(strip_thinking("<think>still thinking"), "")


class HFClientTests(unittest.TestCase):
    def test_reply_contains_only_new_tokens_and_usage(self):
        client, log = make_client()
        completion = client.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})
        self.assertEqual(completion.text, "hij")
        self.assertEqual(completion.usage, {"prompt_tokens": 4, "completion_tokens": 3})
        self.assertEqual(client.name, "hf:stub/model")

    def test_thinking_is_off_by_default_and_configurable(self):
        client, log = make_client()
        client.complete([], {})
        self.assertIs(log["template_calls"][0]["enable_thinking"], False)
        client, log = make_client({"hf": {"enable_thinking": True}})
        client.complete([], {})
        self.assertIs(log["template_calls"][0]["enable_thinking"], True)

    def test_sampling_settings_follow_config(self):
        client, log = make_client()
        client.complete([], {})
        call = log["generate_calls"][0]
        self.assertTrue(call["do_sample"])
        self.assertEqual((call["temperature"], call["top_p"], call["top_k"], call["max_new_tokens"]),
                         (0.7, 0.8, 20, 50))

    def test_zero_or_missing_temperature_is_greedy(self):
        for temperature in (0, None):
            client, log = make_client({"temperature": temperature})
            client.complete([], {})
            call = log["generate_calls"][0]
            self.assertFalse(call["do_sample"])
            self.assertNotIn("temperature", call)

    def test_cpu_load_has_no_device_map_and_gpu_load_shards(self):
        _, cpu_log = make_client(gpu=False)
        _, gpu_log = make_client(gpu=True, cuda_available=True)
        cpu_kwargs = next(entry for entry in cpu_log["load_calls"] if entry[0] == "model")[2]
        gpu_kwargs = next(entry for entry in gpu_log["load_calls"] if entry[0] == "model")[2]
        self.assertNotIn("device_map", cpu_kwargs)
        self.assertEqual(gpu_kwargs["device_map"], "auto")
        self.assertEqual(gpu_kwargs["dtype"], "auto")

    def test_declared_architecture_class_is_used(self):
        _, log = make_client()
        self.assertEqual([entry[0] for entry in log["load_calls"]], ["tokenizer", "model"])

    def test_gpu_flag_without_cuda_stops_with_a_clear_message(self):
        modules, _ = build_stubs(cuda_available=False)
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaises(SystemExit) as raised:
                HFLocalClient({"hf": {"model": "stub/model"}}, gpu=True, seed=1)
        self.assertIn("--gpu", str(raised.exception))

    def test_missing_packages_give_install_hint(self):
        with mock.patch.dict(sys.modules, {"torch": None, "transformers": None}):
            with self.assertRaises(SystemExit) as raised:
                HFLocalClient({"hf": {"model": "stub/model"}}, gpu=False, seed=1)
        self.assertIn("requirements.txt", str(raised.exception))

    def test_generation_failure_becomes_model_error(self):
        client, _ = make_client(generate_error=RuntimeError("CUDA out of memory"))
        with self.assertRaises(ModelError):
            client.complete([], {})

    def test_client_is_marked_serial_so_runner_uses_one_worker(self):
        client, _ = make_client()
        self.assertTrue(client.serial)

    def test_build_client_applies_model_override(self):
        modules, log = build_stubs()
        with mock.patch.dict(sys.modules, modules):
            client = build_client({"hf": {"model": "stub/default"}}, "hf", 1, gpu=False,
                                  hf_model="stub/override")
        self.assertEqual(client.name, "hf:stub/override")


if __name__ == "__main__":
    unittest.main()
