"""Tests for the vLLM backend that run without vLLM or a GPU.

A small stand-in program plays the part of ``vllm serve``: it parses the same command line,
answers ``GET /health`` and ``POST /v1/chat/completions``, and records what it was given. This
checks our own logic: the command line we build, the request body we send, waiting for
readiness, failure handling, and that the server is stopped. It does not test vLLM itself.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.request
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mas_sim.llm import (DEFAULT_VLLM_MODEL, BACKENDS, ModelError, VLLMClient,  # noqa: E402
                         build_client, resolve_backend)

STUB = textwrap.dedent('''
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    argv = sys.argv[1:]
    record = os.environ["STUB_RECORD"]
    with open(record, "w") as handle:
        json.dump({"argv": argv, "pid": os.getpid()}, handle)
    if os.environ.get("STUB_EXIT"):
        print("stub: simulated start-up failure: out of memory", flush=True)
        sys.exit(3)
    time.sleep(float(os.environ.get("STUB_DELAY", "0")))
    port = int(argv[argv.index("--port") + 1])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(record + ".request", "w") as handle:
                json.dump({"body": body, "auth": self.headers.get("Authorization")}, handle)
            reply = os.environ.get("STUB_REPLY", '{"vote": "C", "rationale": "ok"}')
            payload = json.dumps({"choices": [{"message": {"content": reply}}],
                                  "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload.encode())

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
''')


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class VLLMTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.stub = os.path.join(self.dir.name, "stub_vllm.py")
        with open(self.stub, "w") as handle:
            handle.write(STUB)
        self.record = os.path.join(self.dir.name, "record.json")
        patcher = mock.patch.dict(os.environ, {"STUB_RECORD": self.record})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clients = []

    def settings(self, **vllm):
        base = {"command": [sys.executable, self.stub], "startup_poll_seconds": 0.05,
                "startup_timeout_seconds": 30, "stop_timeout_seconds": 5}
        base.update(vllm)
        return {"temperature": 0.7, "max_tokens": 120, "vllm": base}

    def start(self, seed=7, **vllm):
        client = VLLMClient(self.settings(**vllm), seed=seed, log_dir=self.dir.name)
        self.clients.append(client)
        self.addCleanup(client.close)
        return client

    def launched(self):
        with open(self.record) as handle:
            return json.load(handle)

    def request(self):
        with open(self.record + ".request") as handle:
            return json.load(handle)


class BackendSelectionTests(unittest.TestCase):
    def test_vllm_is_a_backend(self):
        self.assertIn("vllm", BACKENDS)
        self.assertEqual(resolve_backend("vllm"), "vllm")
        self.assertEqual(resolve_backend("api", backend="vllm"), "vllm")

    def test_vllm_flags_imply_vllm(self):
        self.assertEqual(resolve_backend("api", vllm_model="x/y"), "vllm")
        self.assertEqual(resolve_backend("hf", vllm_url="http://h:1"), "vllm")
        self.assertEqual(resolve_backend(None, tensor_parallel_size=2), "vllm")

    def test_conflicts_are_rejected(self):
        bad = [dict(backend="hf", vllm_model="x"), dict(backend="api", vllm_url="u"),
               dict(fake=True, tensor_parallel_size=2), dict(backend="vllm", gpu=True),
               dict(backend="vllm", hf_model="m"), dict(gpu=True, vllm_model="x"),
               dict(fake=True, backend="vllm")]
        for kwargs in bad:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                resolve_backend(None, **kwargs)

    def test_existing_hugging_face_rules_are_unchanged(self):
        self.assertEqual(resolve_backend("api", gpu=True), "hf")
        self.assertEqual(resolve_backend("vllm", hf_model="m"), "hf")


class StartupTests(VLLMTestCase):
    def test_command_line_follows_the_config(self):
        self.start(seed=11, model="Org/Model-1", tensor_parallel_size=2, dtype="bfloat16",
                   max_model_len=4096, gpu_memory_utilization=0.8, trust_remote_code=True,
                   extra_args=["--enable-prefix-caching"])
        argv = self.launched()["argv"]
        self.assertEqual(argv[:2], ["serve", "Org/Model-1"])
        for flag, value in (("--host", "127.0.0.1"), ("--dtype", "bfloat16"), ("--seed", "11"),
                            ("--tensor-parallel-size", "2"), ("--max-model-len", "4096"),
                            ("--gpu-memory-utilization", "0.8")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        self.assertIn("--trust-remote-code", argv)
        self.assertEqual(argv[-1], "--enable-prefix-caching")

    def test_optional_flags_are_left_out_when_unset(self):
        self.start(tensor_parallel_size=None, max_model_len=None, gpu_memory_utilization=None)
        argv = self.launched()["argv"]
        for flag in ("--tensor-parallel-size", "--max-model-len", "--gpu-memory-utilization",
                     "--reasoning-parser", "--trust-remote-code"):
            self.assertNotIn(flag, argv)

    def test_default_model_and_free_port(self):
        client = self.start()
        argv = self.launched()["argv"]
        self.assertEqual(argv[1], DEFAULT_VLLM_MODEL)
        port = int(argv[argv.index("--port") + 1])
        self.assertGreater(port, 1023)
        self.assertEqual(client.server_url, f"http://127.0.0.1:{port}")
        self.assertEqual(client.name, f"vllm:{DEFAULT_VLLM_MODEL}")

    def test_reasoning_parser_only_when_thinking_is_on(self):
        self.start(enable_thinking=False, reasoning_parser="qwen3")
        self.assertNotIn("--reasoning-parser", self.launched()["argv"])
        self.start(enable_thinking=True, reasoning_parser="qwen3")
        argv = self.launched()["argv"]
        self.assertEqual(argv[argv.index("--reasoning-parser") + 1], "qwen3")

    def test_waits_until_the_server_answers_health(self):
        with mock.patch.dict(os.environ, {"STUB_DELAY": "0.6"}):
            began = time.monotonic()
            client = self.start()
            self.assertGreaterEqual(time.monotonic() - began, 0.5)
        self.assertTrue(client._healthy())

    def test_server_log_is_written_to_the_log_directory(self):
        self.start()
        self.assertTrue(os.path.exists(os.path.join(self.dir.name, "vllm_server.log")))


class FailureTests(VLLMTestCase):
    def test_server_that_exits_reports_its_log(self):
        with mock.patch.dict(os.environ, {"STUB_EXIT": "1"}):
            with self.assertRaises(SystemExit) as raised:
                VLLMClient(self.settings(), seed=1, log_dir=self.dir.name)
        message = str(raised.exception)
        self.assertIn("exited with code 3", message)
        self.assertIn("out of memory", message)

    def test_startup_timeout_stops_the_server(self):
        with mock.patch.dict(os.environ, {"STUB_DELAY": "60"}):
            with self.assertRaises(SystemExit) as raised:
                VLLMClient(self.settings(startup_timeout_seconds=0.5), seed=1,
                           log_dir=self.dir.name)
        self.assertIn("not ready", str(raised.exception))
        deadline = time.monotonic() + 5
        pid = self.launched()["pid"]
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(process_alive(pid))

    def test_missing_command_gives_an_install_hint(self):
        settings = self.settings(command=["definitely-not-installed-vllm"])
        with self.assertRaises(SystemExit) as raised:
            VLLMClient(settings, seed=1, log_dir=self.dir.name)
        self.assertIn("pip install vllm", str(raised.exception))

    def test_invalid_tensor_parallel_size_is_rejected(self):
        with self.assertRaises(SystemExit):
            VLLMClient(self.settings(tensor_parallel_size=0), seed=1, log_dir=self.dir.name)

    def test_unreachable_server_url_is_reported(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with self.assertRaises(SystemExit) as raised:
            VLLMClient({"vllm": {"base_url": f"http://127.0.0.1:{port}"}}, seed=1)
        self.assertIn("Cannot reach a vLLM server", str(raised.exception))


class RequestTests(VLLMTestCase):
    def test_request_body_and_reply(self):
        client = self.start(model="Org/Model-1", top_p=0.8, top_k=20, repetition_penalty=1.1)
        out = client.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})
        self.assertEqual(json.loads(out.text)["vote"], "C")
        self.assertEqual(out.usage, {"prompt_tokens": 7, "completion_tokens": 3})
        body = self.request()["body"]
        self.assertEqual(body["model"], "Org/Model-1")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["max_tokens"], 120)
        self.assertEqual(body["top_p"], 0.8)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["repetition_penalty"], 1.1)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_thinking_flag_is_sent_to_the_chat_template(self):
        client = self.start(enable_thinking=True)
        client.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})
        self.assertEqual(self.request()["body"]["chat_template_kwargs"], {"enable_thinking": True})

    def test_neutral_penalty_and_null_values_are_not_sent(self):
        client = self.start(repetition_penalty=1.0, top_k=None, top_p=None)
        client.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})
        body = self.request()["body"]
        for key in ("repetition_penalty", "top_k", "top_p"):
            self.assertNotIn(key, body)

    def test_null_temperature_means_greedy(self):
        settings = self.settings()
        settings["temperature"] = None
        client = VLLMClient(settings, seed=1, log_dir=self.dir.name)
        self.addCleanup(client.close)
        client.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})
        self.assertEqual(self.request()["body"]["temperature"], 0)

    def test_placeholder_key_is_sent_unless_one_is_configured(self):
        client = self.start()
        client.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})
        self.assertEqual(self.request()["auth"], "Bearer EMPTY")
        keyed = self.start(api_key="secret-123")
        keyed.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})
        self.assertEqual(self.request()["auth"], "Bearer secret-123")

    def test_reasoning_left_in_the_reply_is_removed(self):
        with mock.patch.dict(os.environ, {"STUB_REPLY": 'I think.</think>{"vote": "A"}'}):
            client = self.start(enable_thinking=True)
            out = client.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})
        self.assertEqual(out.text, '{"vote": "A"}')

    def test_http_errors_become_model_errors(self):
        client = self.start()
        client.close()  # the server is gone, so the request must fail
        client.max_http_retries = 0
        with self.assertRaises(ModelError):
            client.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})


class ExternalServerTests(VLLMTestCase):
    def test_connects_without_starting_anything(self):
        owner = self.start()
        with mock.patch("mas_sim.llm.subprocess.Popen") as popen:
            client = VLLMClient({"temperature": 0, "vllm": {"base_url": owner.server_url + "/v1/"}},
                                seed=1)
        popen.assert_not_called()
        self.assertFalse(client.owns_server)
        self.assertEqual(client.base_url, owner.server_url + "/v1")
        client.complete([{"role": "user", "content": "hi"}], {"call_kind": "final_vote"})
        client.close()
        self.assertTrue(owner._healthy(), "closing a connected client must not stop the server")


class ShutdownTests(VLLMTestCase):
    def test_close_stops_the_server_and_is_repeatable(self):
        client = self.start()
        pid = self.launched()["pid"]
        self.assertTrue(process_alive(pid))
        client.close()
        client.close()
        deadline = time.monotonic() + 5
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(process_alive(pid))
        with self.assertRaises(Exception):
            urllib.request.urlopen(client.server_url + "/health", timeout=1)

    def test_client_runs_trials_in_parallel(self):
        client = self.start()
        self.assertFalse(getattr(client, "serial", False))


class BuildClientTests(VLLMTestCase):
    def test_overrides_from_flags_win_over_config(self):
        settings = {"temperature": 0, "vllm": {"model": "From/Config", "command": [sys.executable, self.stub],
                                               "startup_poll_seconds": 0.05, "stop_timeout_seconds": 5}}
        client = build_client(settings, "vllm", seed=3, vllm_model="From/Flag",
                              tensor_parallel_size=2, log_dir=self.dir.name)
        self.addCleanup(client.close)
        argv = self.launched()["argv"]
        self.assertEqual(argv[1], "From/Flag")
        self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "2")

    def test_url_flag_connects_to_a_running_server(self):
        owner = self.start()
        client = build_client({"temperature": 0, "vllm": {}}, "vllm", seed=3, vllm_url=owner.server_url)
        self.assertFalse(client.owns_server)


class RunnerTests(VLLMTestCase):
    def test_runner_closes_the_server_and_hides_secrets(self):
        from mas_sim import runner
        config = {
            "llm": {"backend": "vllm", "api_key": "top-secret", "temperature": 0,
                    "extra_headers": {"X-Token": "header-secret"},
                    "vllm": {"command": [sys.executable, self.stub], "api_key": "vllm-secret",
                             "startup_poll_seconds": 0.05, "stop_timeout_seconds": 5}},
            "run": {"seed": 1, "items_per_scenario": 1, "repeats": 1, "parallel_trials": 2,
                    "output_dir": "runs"},
            "scenarios": ["payment_approval"],
            "experiment_1_minority_conversion": {"enabled": False},
            "experiment_2_hidden_profile": {"enabled": True, "models": 4, "info_modes": ["full"],
                                            "rounds": [1]},
        }
        path = os.path.join(self.dir.name, "config.json")
        with open(path, "w") as handle:
            json.dump(config, handle)
        reply = {"STUB_REPLY": json.dumps({"private_belief": {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.1},
                                           "group_forecast": {"A": 0.2, "B": 0.2, "C": 0.4, "D": 0.1,
                                                              "Tie": 0.1},
                                           "peer_forecasts": {}})}
        with mock.patch.dict(os.environ, reply):
            run_dir = runner.run(path, limit=1, output_dir=self.dir.name)
        pid = self.launched()["pid"]
        deadline = time.monotonic() + 5
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(process_alive(pid), "the server must be stopped when the run ends")
        self.assertTrue(run_dir.endswith("-vllm"))
        with open(os.path.join(run_dir, "run_config.json")) as handle:
            text = handle.read()
        for secret in ("top-secret", "vllm-secret", "header-secret"):
            self.assertNotIn(secret, text)
        self.assertIn("X-Token", text)
        self.assertEqual(json.loads(text)["backend"], "vllm")


class InterruptionTests(VLLMTestCase):
    def test_sigterm_stops_the_server(self):
        script = textwrap.dedent(f'''
            import sys, time
            sys.path.insert(0, {ROOT!r})
            from mas_sim.llm import VLLMClient
            settings = {{"vllm": {{"command": [sys.executable, {self.stub!r}],
                                  "startup_poll_seconds": 0.05, "stop_timeout_seconds": 5}}}}
            client = VLLMClient(settings, seed=1, log_dir={self.dir.name!r})
            print("ready", flush=True)
            time.sleep(60)
        ''')
        child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        self.addCleanup(child.kill)
        for line in child.stdout:
            if line.strip() == "ready":
                break
        server_pid = self.launched()["pid"]
        self.assertTrue(process_alive(server_pid))
        os.kill(child.pid, signal.SIGTERM)
        child.wait(timeout=20)
        deadline = time.monotonic() + 5
        while process_alive(server_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(process_alive(server_pid), "SIGTERM must not leave the server running")

    def test_sigterm_handler_is_restored_after_close(self):
        before = signal.getsignal(signal.SIGTERM)
        client = self.start()
        self.assertNotEqual(signal.getsignal(signal.SIGTERM), before)
        client.close()
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_interrupt_drops_queued_trials_and_closes_the_client(self):
        from mas_sim import runner
        from mas_sim.llm import FakeClient

        class Counting(FakeClient):
            calls = 0
            closed = False

            def complete(self, messages, spec):
                Counting.calls += 1
                return super().complete(messages, spec)

            def close(self):
                Counting.closed = True

        config = {
            "llm": {"backend": "fake"},
            "run": {"seed": 1, "items_per_scenario": 1, "repeats": 1, "parallel_trials": 1,
                    "output_dir": "runs"},
            "scenarios": ["payment_approval"],
            "experiment_1_minority_conversion": {
                "enabled": True, "compositions": [[4, 2]], "rounds": [1],
                "arms": ["consistent_wrong", "truthful_sham", "no_script"]},
            "experiment_2_hidden_profile": {"enabled": False},
        }
        path = os.path.join(self.dir.name, "config.json")
        with open(path, "w") as handle:
            json.dump(config, handle)
        with mock.patch.object(runner, "build_client", return_value=Counting(seed=1)), \
                mock.patch.object(runner, "as_completed", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                runner.run(path, fake=True, output_dir=self.dir.name)
        self.assertTrue(Counting.closed)
        self.assertLess(Counting.calls, 3 * 24, "queued trials should have been cancelled")


if __name__ == "__main__":
    unittest.main()
