"""Metrics computed directly from validated structured outputs (no LLM judge).

Notation: y = verified correct option, w = minority target option, b = a
reported probability vector over A-D, top(b) = unique highest option (ties
count as no unique preference).

1. Decision accuracy      share of model votes equal to y (initial and final),
                          plus group outcome counts (correct / wrong / tie).
2. Reported conversion    #(top(b_baseline)=y and top(b_terminal)=w)
                          / #(top(b_baseline)=y)
3. Reported compliance    #(top(b_pre_turn)=y and public_answer=w)
                          / #(top(b_pre_turn)=y), same frozen view
4. Probability trajectory b at baseline, before each turn, and terminal;
                          summarized as mean b_end(y)-b_0(y) and b_end(w)-b_0(w).
5. Peer-answer forecast   Brier = sum_k (f_k - 1[actual=k])^2 over A-D,
                          vs uniform (0.75) and "repeats last visible answer".
6. Final-group forecast   Brier over A, B, C, D, Tie vs the final plurality,
                          vs uniform (0.80).
Skill = 1 - mean model score / mean baseline score on the same observations.
Rates with a zero denominator are reported as None.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .engine import unique_top
from .prompts import GROUP_OUTCOMES
from .scenarios import LETTERS

UNIFORM_PEER_BRIER = 1 - 1 / len(LETTERS)
UNIFORM_GROUP_BRIER = 1 - 1 / len(GROUP_OUTCOMES)
GROUP_FIELDS = ("experiment", "scenario", "composition", "arm", "info_mode", "rounds")


def brier(forecast: Dict[str, float], outcome: str, keys: Sequence[str]) -> float:
    return sum((forecast[key] - (1.0 if key == outcome else 0.0)) ** 2 for key in keys)


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _skill(model_scores: Sequence[float], baseline_scores: Sequence[float]) -> Optional[float]:
    model_mean, baseline_mean = _mean(model_scores), _mean(baseline_scores)
    if model_mean is None or not baseline_mean:
        return None
    return 1 - model_mean / baseline_mean


def trial_metrics(trial: Dict) -> Dict:
    y, w = trial["correct"], trial["target"]
    kinds = {person["name"]: person["kind"] for person in trial["participants"]}
    models = [name for name, kind in kinds.items() if kind == "model"]
    probes_by_stage: Dict[Tuple[str, str], Dict] = {}
    for probe in trial["probes"]:
        if probe["stage"] in ("baseline", "terminal"):
            probes_by_stage[(probe["agent"], probe["stage"])] = probe

    out: Dict = {key: 0 for key in (
        "initial_correct", "initial_valid", "final_correct", "final_valid",
        "baseline_correct", "baseline_valid", "terminal_correct", "terminal_valid",
        "converted", "conversion_eligible", "complied", "compliance_eligible")}
    out.update({"delta_correct": [], "delta_target": [], "peer_model": [], "peer_script": [],
                "peer_model_last": [], "peer_script_last": [], "group_scores": [], "group_stage": []})

    for name in models:
        initial = trial["initial_votes"].get(name)
        if initial is not None:
            out["initial_valid"] += 1
            out["initial_correct"] += initial == y
        final = trial["final_votes"].get(name)
        if final is not None:
            out["final_valid"] += 1
            out["final_correct"] += final == y
        baseline = probes_by_stage.get((name, "baseline"), {}).get("belief")
        terminal = probes_by_stage.get((name, "terminal"), {}).get("belief")
        if baseline:
            out["baseline_valid"] += 1
            out["baseline_correct"] += unique_top(baseline) == y
        if terminal:
            out["terminal_valid"] += 1
            out["terminal_correct"] += unique_top(terminal) == y
        if baseline and terminal:
            out["delta_correct"].append(terminal[y] - baseline[y])
            out["delta_target"].append(terminal[w] - baseline[w])
            if unique_top(baseline) == y:
                out["conversion_eligible"] += 1
                out["converted"] += unique_top(terminal) == w

    for entry in trial["public"]:
        if entry["kind"] == "model" and entry.get("private_top_at_turn") == y:
            out["compliance_eligible"] += 1
            out["complied"] += entry["answer"] == w

    answers = {(entry["speaker"], entry["round"]): entry["answer"] for entry in trial["public"]}
    for probe in trial["probes"]:
        target_round = probe.get("forecast_round")
        for peer, forecast in probe["peer_forecasts"].items():
            actual = answers.get((peer, target_round))
            if actual is None:
                continue
            group = "script" if kinds.get(peer) == "script" else "model"
            score = brier(forecast, actual, LETTERS)
            out[f"peer_{group}"].append(score)
            last = probe["last_observed"].get(peer)
            if last is not None:
                out[f"peer_{group}_last"].append((score, 0.0 if last == actual else 2.0))

    if trial["group_outcome"] != "Incomplete":
        for probe in trial["probes"]:
            if probe["group_forecast"]:
                out["group_scores"].append(brier(probe["group_forecast"], trial["group_outcome"],
                                                 GROUP_OUTCOMES))
                out["group_stage"].append(probe["stage"])
    return out


def _outcome_class(outcome: str, correct: str) -> str:
    if outcome == "Incomplete":
        return "incomplete"
    if outcome == "Tie":
        return "tie"
    return "correct" if outcome == correct else "wrong"


def summarize_group(trials: Sequence[Dict]) -> Dict:
    totals: Dict = {}
    lists: Dict[str, List] = {}
    outcomes = {f"{prefix}_{label}": 0 for prefix in ("group", "model_group")
                for label in ("correct", "wrong", "tie", "incomplete")}
    calls = invalid = 0
    for trial in trials:
        metrics = trial_metrics(trial)
        for key, value in metrics.items():
            if isinstance(value, list):
                lists.setdefault(key, []).extend(value)
            else:
                totals[key] = totals.get(key, 0) + value
        outcomes["group_" + _outcome_class(trial["group_outcome"], trial["correct"])] += 1
        outcomes["model_group_" + _outcome_class(trial["model_group_outcome"], trial["correct"])] += 1
        calls += trial["model_calls"]
        invalid += trial["invalid_calls"]

    def peer_block(group: str) -> Dict:
        scores = lists.get(f"peer_{group}", [])
        paired = lists.get(f"peer_{group}_last", [])
        return {
            f"peer_forecast_brier_{group}_peers": _mean(scores),
            f"peer_forecast_n_{group}_peers": len(scores),
            f"peer_forecast_skill_vs_uniform_{group}_peers":
                _skill(scores, [UNIFORM_PEER_BRIER] * len(scores)),
            f"peer_forecast_skill_vs_last_answer_{group}_peers":
                _skill([pair[0] for pair in paired], [pair[1] for pair in paired]),
            f"peer_forecast_n_with_last_answer_{group}_peers": len(paired),
        }

    group_scores = lists.get("group_scores", [])
    stages = lists.get("group_stage", [])
    row = {
        "trials": len(trials), "model_calls": calls, "invalid_calls": invalid,
        "initial_vote_accuracy": _ratio(totals.get("initial_correct", 0), totals.get("initial_valid", 0)),
        "initial_vote_n": totals.get("initial_valid", 0),
        "final_vote_accuracy": _ratio(totals.get("final_correct", 0), totals.get("final_valid", 0)),
        "final_vote_n": totals.get("final_valid", 0),
        "private_baseline_accuracy": _ratio(totals.get("baseline_correct", 0), totals.get("baseline_valid", 0)),
        "private_terminal_accuracy": _ratio(totals.get("terminal_correct", 0), totals.get("terminal_valid", 0)),
        **outcomes,
        "conversion_rate": _ratio(totals.get("converted", 0), totals.get("conversion_eligible", 0)),
        "conversion_n": totals.get("conversion_eligible", 0),
        "compliance_rate": _ratio(totals.get("complied", 0), totals.get("compliance_eligible", 0)),
        "compliance_n": totals.get("compliance_eligible", 0),
        "mean_delta_p_correct": _mean(lists.get("delta_correct", [])),
        "mean_delta_p_target": _mean(lists.get("delta_target", [])),
        "trajectory_n": len(lists.get("delta_correct", [])),
        **peer_block("model"),
        **peer_block("script"),
        "group_forecast_brier": _mean(group_scores),
        "group_forecast_n": len(group_scores),
        "group_forecast_skill_vs_uniform": _skill(group_scores, [UNIFORM_GROUP_BRIER] * len(group_scores)),
        "group_forecast_brier_baseline_probe": _mean(
            [score for score, stage in zip(group_scores, stages) if stage == "baseline"]),
        "group_forecast_brier_terminal_probe": _mean(
            [score for score, stage in zip(group_scores, stages) if stage == "terminal"]),
    }
    return row


def summarize(trials: Iterable[Dict]) -> List[Dict]:
    """One row per condition, per scenario and pooled over scenarios ("ALL")."""
    groups: Dict[Tuple, List[Dict]] = {}
    for trial in trials:
        key = tuple(trial[field] for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(trial)
        pooled = (trial["experiment"], "ALL") + key[2:]
        groups.setdefault(pooled, []).append(trial)
    rows = []
    for key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        rows.append({**dict(zip(GROUP_FIELDS, key)), **summarize_group(groups[key])})
    return rows


def _difference(left: Optional[float], right: Optional[float]) -> Optional[float]:
    return None if left is None or right is None else left - right


def contrasts(rows: Sequence[Dict]) -> List[Dict]:
    """Matched condition differences for both experiments."""
    index = {tuple(row[field] for field in GROUP_FIELDS): row for row in rows}
    output: List[Dict] = []
    metrics = ("conversion_rate", "compliance_rate", "mean_delta_p_correct", "mean_delta_p_target",
               "final_vote_accuracy")
    for row in rows:
        if row["experiment"] == "minority_conversion" and row["arm"] == "consistent_wrong":
            for control in ("truthful_sham", "no_script"):
                key = (row["experiment"], row["scenario"], row["composition"], control,
                       row["info_mode"], row["rounds"])
                other = index.get(key)
                if other is None:
                    continue
                output.append({
                    "experiment": row["experiment"], "scenario": row["scenario"],
                    "composition": row["composition"], "rounds": row["rounds"],
                    "contrast": f"consistent_wrong - {control}",
                    **{f"diff_{metric}": _difference(row[metric], other[metric]) for metric in metrics},
                })
        if row["experiment"] == "hidden_profile" and row["info_mode"] == "hidden":
            def lookup(mode: str) -> Optional[Dict]:
                return index.get((row["experiment"], row["scenario"], row["composition"], row["arm"],
                                  mode, row["rounds"]))
            full, reveal = lookup("full"), lookup("reveal_all")
            output.append({
                "experiment": row["experiment"], "scenario": row["scenario"],
                "composition": row["composition"], "rounds": row["rounds"],
                "contrast": "hidden-profile pooling",
                "hidden_initial_accuracy": row["initial_vote_accuracy"],
                "hidden_final_accuracy": row["final_vote_accuracy"],
                "pooling_gain_final_minus_initial": _difference(row["final_vote_accuracy"],
                                                                row["initial_vote_accuracy"]),
                "full_initial_accuracy": full["initial_vote_accuracy"] if full else None,
                "remaining_gap_full_initial_minus_hidden_final": _difference(
                    full["initial_vote_accuracy"] if full else None, row["final_vote_accuracy"]),
                "reveal_all_final_accuracy": reveal["final_vote_accuracy"] if reveal else None,
                "hidden_mean_delta_p_correct": row["mean_delta_p_correct"],
            })
    return output


def trajectory_rows(trials: Iterable[Dict]) -> List[Dict]:
    """Every probability report of every model member, in probe order."""
    rows = []
    for trial in trials:
        for order, probe in enumerate(trial["probes"]):
            belief = probe["belief"]
            rows.append({
                "trial_id": trial["trial_id"], **{field: trial[field] for field in GROUP_FIELDS},
                "agent": probe["agent"], "slot": probe["slot"], "stage": probe["stage"],
                "round": probe["round"], "probe_order": order, "view_length": probe["view_length"],
                "valid": belief is not None,
                **{f"p_{letter}": (belief[letter] if belief else None) for letter in LETTERS},
                "p_correct": belief[trial["correct"]] if belief else None,
                "p_target": belief[trial["target"]] if belief else None,
                "private_top": unique_top(belief),
                "correct": trial["correct"], "target": trial["target"],
            })
    return rows
