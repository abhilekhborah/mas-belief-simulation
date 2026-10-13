"""Loads the config, runs every trial, and writes one self-contained results file per run.

Each run gets its own folder ``runs/<start time>_<backend>/`` holding ``results.json`` (everything:
run metadata, scores, every trial with its prompts and raw replies) plus three flat CSV tables.
``runs/history.jsonl`` gets one line per run, so past runs can be listed and compared.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from . import metrics
from .engine import TrialRunner
from .experiments import build_trials, planned_calls
from .llm import build_client, resolve_backend
from .scenarios import verify_all

SCHEMA_VERSION = 1
ENV_OVERRIDES = {"MAS_BASE_URL": "base_url", "MAS_API_KEY": "api_key", "MAS_MODEL": "model"}
HEADLINE_METRICS = ("trials", "initial_vote_accuracy", "final_vote_accuracy", "conversion_rate",
                    "compliance_rate", "mean_delta_p_correct", "mean_delta_p_target",
                    "group_forecast_brier")


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


def _write_json_atomically(path: str, document: Dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _redacted_llm(llm: Dict) -> Dict:
    """Copy of the llm settings that is safe to write to disk: no keys, no header values."""
    safe = {key: value for key, value in llm.items() if key != "api_key"}
    if isinstance(safe.get("vllm"), dict):
        safe["vllm"] = {key: value for key, value in safe["vllm"].items() if key != "api_key"}
    if safe.get("extra_headers"):
        safe["extra_headers"] = {name: "<hidden>" for name in safe["extra_headers"]}
    return safe


def describe_plan(config: Dict) -> Dict:
    trials = build_trials(config)
    by_experiment: Dict[str, Dict[str, int]] = {}
    for trial in trials:
        entry = by_experiment.setdefault(trial.experiment, {"trials": 0, "calls": 0})
        entry["trials"] += 1
        entry["calls"] += planned_calls(trial)
    return {"trials": len(trials), "calls_without_retries": sum(map(planned_calls, trials)),
            "by_experiment": by_experiment}


def _stop_on_sigterm():
    """Turn SIGTERM into a normal exit so results are still saved; returns an undo function."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    previous = signal.getsignal(signal.SIGTERM)
    if previous != signal.SIG_DFL:
        return lambda: None

    def handler(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)
    return lambda: signal.signal(signal.SIGTERM, previous)


def _unused_folder(path: str) -> str:
    candidate, number = path, 2
    while os.path.exists(candidate):
        candidate, number = f"{path}_{number}", number + 1
    return candidate


def run(config_path: str, limit: Optional[int] = None, output_dir: Optional[str] = None,
        backend: Optional[str] = None, gpu: bool = False, hf_model: Optional[str] = None,
        vllm_model: Optional[str] = None, vllm_url: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None) -> str:
    verify_all()
    config = load_config(config_path)
    settings = {**config["llm"], **config["run"]}
    trials = build_trials(config)
    if limit is not None:
        trials = trials[:limit]
    chosen = resolve_backend(config["llm"].get("backend"), backend, gpu, hf_model, vllm_model,
                             vllm_url, tensor_parallel_size)

    started = datetime.now().astimezone()
    base = output_dir or os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                      config["run"].get("output_dir", "runs"))
    run_dir = _unused_folder(os.path.join(base, f"{started:%Y%m%dT%H%M%S}_{chosen}"))
    restore_sigterm = _stop_on_sigterm()
    client = None
    try:
        client = build_client(config["llm"], chosen, int(config["run"]["seed"]), gpu=gpu,
                              hf_model=hf_model, vllm_model=vllm_model, vllm_url=vllm_url,
                              tensor_parallel_size=tensor_parallel_size, log_dir=run_dir)
        return _execute(client, config, settings, trials, chosen, gpu, base, run_dir, started)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()  # stops a vLLM server that this run started
        restore_sigterm()


def _execute(client, config: Dict, settings: Dict, trials: List, chosen: str, gpu: bool,
             base: str, run_dir: str, started: datetime) -> str:
    os.makedirs(run_dir, exist_ok=True)
    partial_path = os.path.join(run_dir, "trials.partial.jsonl")
    order = {trial.trial_id: index for index, trial in enumerate(trials)}
    results: List[Dict] = []
    failures: List[Dict] = []
    status = "completed"
    workers = max(1, int(config["run"].get("parallel_trials", 1)))
    if getattr(client, "serial", False):
        workers = 1  # a single local model generates one reply at a time
    print(f"Running {len(trials)} trials with {client.name} -> {run_dir}")
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(TrialRunner(trial, client, settings).run): trial for trial in trials}
        for done, future in enumerate(as_completed(futures), start=1):
            trial = futures[future]
            try:
                record = future.result()
            except Exception as error:  # keep other trials running; record the failure
                failures.append({"trial_id": trial.trial_id, "error": repr(error)})
                continue
            record["scores"] = metrics.summarize_group([record])
            results.append(record)
            with open(partial_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if done % 10 == 0 or done == len(trials):
                print(f"  {done}/{len(trials)} trials finished")
    except (KeyboardInterrupt, SystemExit):
        status = "interrupted"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)  # on Ctrl-C, queued trials are dropped
        results.sort(key=lambda record: order[record["trial_id"]])
        _save(client, config, chosen, gpu, base, run_dir, started, status, len(trials), results,
              failures)
        if os.path.exists(partial_path):
            os.remove(partial_path)
    return run_dir


def _save(client, config: Dict, chosen: str, gpu: bool, base: str, run_dir: str,
          started: datetime, status: str, planned: int, results: List[Dict],
          failures: List[Dict]) -> None:
    finished = datetime.now().astimezone()
    summary_rows = metrics.summarize(results)
    contrast_rows = metrics.contrasts(summary_rows)
    usage: Dict[str, int] = {}
    for record in results:
        for key, value in record["usage"].items():
            usage[key] = usage.get(key, 0) + value
    run_id = os.path.basename(run_dir)
    model = getattr(client, "model_name", None) or getattr(client, "model", None)
    meta = {
        "run_id": run_id, "status": status,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_seconds": round((finished - started).total_seconds(), 1),
        "backend": chosen, "client": client.name, "model": str(model) if model else None,
        "gpu": gpu, "command": sys.argv,
        "trials_planned": planned, "trials_completed": len(results),
        "trials_failed": len(failures),
        "model_calls": sum(record["model_calls"] for record in results),
        "invalid_calls": sum(record["invalid_calls"] for record in results),
        "usage": usage,
        "config": {"llm": _redacted_llm(config["llm"]), "run": config["run"],
                   "scenarios": config["scenarios"],
                   "experiment_1_minority_conversion": config.get("experiment_1_minority_conversion"),
                   "experiment_2_hidden_profile": config.get("experiment_2_hidden_profile")},
    }
    _write_json_atomically(os.path.join(run_dir, "results.json"), {
        "schema_version": SCHEMA_VERSION, "run": meta, "summary": summary_rows,
        "contrasts": contrast_rows, "failed_trials": failures, "trials": results})
    _write_csv(os.path.join(run_dir, "summary.csv"), summary_rows)
    _write_csv(os.path.join(run_dir, "contrasts.csv"), contrast_rows)
    _write_csv(os.path.join(run_dir, "trajectories.csv"), metrics.trajectory_rows(results))

    headline: Dict[str, Dict] = {}
    for experiment in sorted({record["experiment"] for record in results}):
        row = metrics.summarize_group([r for r in results if r["experiment"] == experiment])
        headline[experiment] = {key: row[key] for key in HEADLINE_METRICS}
    entry = {key: meta[key] for key in ("run_id", "status", "started_at", "finished_at",
                                        "duration_seconds", "backend", "model", "trials_planned",
                                        "trials_completed", "trials_failed", "model_calls",
                                        "invalid_calls")}
    entry.update({"results_file": f"{run_id}/results.json", "headline": headline})
    with open(os.path.join(base, "history.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Run {status}: {len(results)} of {planned} trials completed, {len(failures)} failed. "
          f"Results: {os.path.join(run_dir, 'results.json')}")
