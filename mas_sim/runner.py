"""Loads config, runs every trial, and writes metrics and raw records."""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

from . import metrics
from .engine import EventSink, TrialRunner
from .experiments import build_trials, planned_calls
from .llm import build_client, resolve_backend
from .scenarios import verify_all

ENV_OVERRIDES = {"MAS_BASE_URL": "base_url", "MAS_API_KEY": "api_key", "MAS_MODEL": "model"}


def load_config(path: str) -> Dict:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    for variable, key in ENV_OVERRIDES.items():
        if os.environ.get(variable):
            config["llm"][key] = os.environ[variable]
    return config


def _write_csv(path: str, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def describe_plan(config: Dict) -> Dict:
    trials = build_trials(config)
    by_experiment: Dict[str, Dict[str, int]] = {}
    for trial in trials:
        entry = by_experiment.setdefault(trial.experiment, {"trials": 0, "calls": 0})
        entry["trials"] += 1
        entry["calls"] += planned_calls(trial)
    return {"trials": len(trials), "calls_without_retries": sum(map(planned_calls, trials)),
            "by_experiment": by_experiment}


def run(config_path: str, fake: bool = False, limit: Optional[int] = None,
        output_dir: Optional[str] = None, backend: Optional[str] = None, gpu: bool = False,
        hf_model: Optional[str] = None, vllm_model: Optional[str] = None,
        vllm_url: Optional[str] = None, tensor_parallel_size: Optional[int] = None) -> str:
    verify_all()
    config = load_config(config_path)
    settings = {**config["llm"], **config["run"]}
    trials = build_trials(config)
    if limit is not None:
        trials = trials[:limit]
    chosen = resolve_backend(config["llm"].get("backend"), fake, backend, gpu, hf_model,
                             vllm_model, vllm_url, tensor_parallel_size)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    base = output_dir or os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                      config["run"].get("output_dir", "runs"))
    suffix = {"fake": "-fake", "hf": "-hf", "vllm": "-vllm"}.get(chosen, "")
    run_dir = os.path.join(base, f"{stamp}{suffix}")
    client = build_client(config["llm"], chosen, int(config["run"]["seed"]), gpu=gpu,
                          hf_model=hf_model, vllm_model=vllm_model, vllm_url=vllm_url,
                          tensor_parallel_size=tensor_parallel_size, log_dir=run_dir)
    try:
        return _run_trials(client, config, settings, trials, chosen, gpu, run_dir)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()  # stops a vLLM server that this run started


def _redacted_llm(llm: Dict) -> Dict:
    """Copy of the llm settings that is safe to write to disk: no keys, no header values."""
    safe = {key: value for key, value in llm.items() if key != "api_key"}
    if isinstance(safe.get("vllm"), dict):
        safe["vllm"] = {key: value for key, value in safe["vllm"].items() if key != "api_key"}
    if safe.get("extra_headers"):
        safe["extra_headers"] = {name: "<hidden>" for name in safe["extra_headers"]}
    return safe


def _run_trials(client, config: Dict, settings: Dict, trials: List, chosen: str, gpu: bool,
                run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    safe_llm = _redacted_llm(config["llm"])
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"llm": safe_llm, "run": config["run"], "backend": chosen, "gpu": gpu,
                   "client": client.name, "trials": len(trials)}, handle, indent=2)

    sink = EventSink(os.path.join(run_dir, "calls.jsonl"))
    trials_path = os.path.join(run_dir, "trials.jsonl")
    results: List[Dict] = []
    failures: List[Dict] = []
    workers = max(1, int(config["run"].get("parallel_trials", 1)))
    if getattr(client, "serial", False):
        workers = 1  # a single local model generates one reply at a time
    print(f"Running {len(trials)} trials with {client.name} -> {run_dir}")
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(TrialRunner(trial, client, settings, sink).run): trial for trial in trials}
        for done, future in enumerate(as_completed(futures), start=1):
            trial = futures[future]
            try:
                record = future.result()
            except Exception as error:  # keep other trials running; record the failure
                failures.append({"trial_id": trial.trial_id, "error": repr(error)})
                continue
            results.append(record)
            with open(trials_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if done % 10 == 0 or done == len(trials):
                print(f"  {done}/{len(trials)} trials finished")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)  # on Ctrl-C, queued trials are dropped

    summary_rows = metrics.summarize(results)
    contrast_rows = metrics.contrasts(summary_rows)
    _write_csv(os.path.join(run_dir, "summary.csv"), summary_rows)
    _write_csv(os.path.join(run_dir, "contrasts.csv"), contrast_rows)
    _write_csv(os.path.join(run_dir, "trajectories.csv"), metrics.trajectory_rows(results))
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary_rows, "contrasts": contrast_rows, "failed_trials": failures},
                  handle, indent=2)
    print(f"Done: {len(results)} trials completed, {len(failures)} failed. Outputs in {run_dir}")
    return run_dir
