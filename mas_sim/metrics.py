"""Threshold-free reported-answer metrics and role-separated forecast scores.

Accuracy, answer changes, and probability movement use honest agents only.
Forecasts are scored separately by forecaster and peer role. Shared source
trials are referenced by each analysis condition, never duplicated within a
condition's sample. Empty denominators produce None, not zero.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .engine import unique_top
from .prompts import GROUP_OUTCOMES
from .scenarios import LETTERS

UNIFORM_PEER_BRIER = 1 - 1 / len(LETTERS)
UNIFORM_GROUP_BRIER = 1 - 1 / len(GROUP_OUTCOMES)
GROUP_FIELDS = ("experiment", "scenario", "composition", "arm", "info_mode", "rounds",
                "order_condition")


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
    honest = [name for name, kind in kinds.items() if kind == "honest"]
    endpoints = {(entry["agent"], entry["stage"]): entry for entry in trial["decisions"]
                 if entry["stage"] in ("initial", "final")}

    out: Dict = {key: 0 for key in (
        "initial_correct", "initial_valid", "final_correct", "final_valid",
        "paired_answers", "changed_answers", "switch_eligible", "switched_to_target")}
    out.update({"delta_correct": [], "delta_target": []})
    for role in ("honest", "confederate"):
        out[f"group_{role}"] = []
        for peer_role in ("honest", "confederate"):
            out[f"peer_{role}_{peer_role}"] = []
            out[f"peer_{role}_{peer_role}_last"] = []

    for name in honest:
        initial = trial["initial_votes"].get(name)
        if initial is not None:
            out["initial_valid"] += 1
            out["initial_correct"] += initial == y
        final = trial["final_votes"].get(name)
        if final is not None:
            out["final_valid"] += 1
            out["final_correct"] += final == y
        if initial is not None and final is not None:
            out["paired_answers"] += 1
            out["changed_answers"] += initial != final
            if initial == y:
                out["switch_eligible"] += 1
                out["switched_to_target"] += final == w
        baseline = endpoints.get((name, "initial"), {}).get("probabilities")
        terminal = endpoints.get((name, "final"), {}).get("probabilities")
        if baseline and terminal:
            out["delta_correct"].append(terminal[y] - baseline[y])
            out["delta_target"].append(terminal[w] - baseline[w])

    answers = {(entry["speaker"], entry["round"]): entry["answer"] for entry in trial["public"]}
    for record in trial["forecasts"]:
        role = kinds[record["agent"]]
        target_round = record["forecast_round"]
        for peer, forecast in record["peer_forecasts"].items():
            actual = answers.get((peer, target_round))
            if actual is None:
                continue
            group = f"{role}_{kinds[peer]}"
            score = brier(forecast, actual, LETTERS)
            out[f"peer_{group}"].append(score)
            last = record["last_observed"].get(peer)
            if last is not None:
                out[f"peer_{group}_last"].append((score, 0.0 if last == actual else 2.0))
        if trial["group_outcome"] != "Incomplete" and record["group_forecast"]:
            out[f"group_{role}"].append(brier(record["group_forecast"], trial["group_outcome"],
                                              GROUP_OUTCOMES))
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
    outcomes = {f"{prefix}_{label}": 0 for prefix in ("group", "honest_group")
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
        outcomes["honest_group_" + _outcome_class(trial["honest_group_outcome"], trial["correct"])] += 1
        calls += trial["model_calls"]
        invalid += trial["invalid_calls"]

    def peer_block(role: str, group: str) -> Dict:
        prefix = "" if role == "honest" else "confederate_"
        scores = lists.get(f"peer_{role}_{group}", [])
        paired = lists.get(f"peer_{role}_{group}_last", [])
        return {
            f"{prefix}peer_forecast_brier_{group}_peers": _mean(scores),
            f"{prefix}peer_forecast_n_{group}_peers": len(scores),
            f"{prefix}peer_forecast_skill_vs_uniform_{group}_peers":
                _skill(scores, [UNIFORM_PEER_BRIER] * len(scores)),
            f"{prefix}peer_forecast_skill_vs_last_answer_{group}_peers":
                _skill([pair[0] for pair in paired], [pair[1] for pair in paired]),
            f"{prefix}peer_forecast_n_with_last_answer_{group}_peers": len(paired),
        }

    def group_block(role: str) -> Dict:
        prefix = "" if role == "honest" else "confederate_"
        scores = lists.get(f"group_{role}", [])
        return {f"{prefix}group_forecast_brier": _mean(scores),
                f"{prefix}group_forecast_n": len(scores),
                f"{prefix}group_forecast_skill_vs_uniform":
                    _skill(scores, [UNIFORM_GROUP_BRIER] * len(scores))}

    row = {
        "trials": len(trials), "model_calls": calls, "invalid_calls": invalid,
        "initial_vote_accuracy": _ratio(totals.get("initial_correct", 0), totals.get("initial_valid", 0)),
        "initial_vote_n": totals.get("initial_valid", 0),
        "final_vote_accuracy": _ratio(totals.get("final_correct", 0), totals.get("final_valid", 0)),
        "final_vote_n": totals.get("final_valid", 0),
        **outcomes,
        "answer_change_rate": _ratio(totals.get("changed_answers", 0), totals.get("paired_answers", 0)),
        "answer_change_n": totals.get("paired_answers", 0),
        "answer_switch_to_target_rate": _ratio(totals.get("switched_to_target", 0),
                               totals.get("switch_eligible", 0)),
        "answer_switch_to_target_n": totals.get("switch_eligible", 0),
        "mean_delta_p_correct": _mean(lists.get("delta_correct", [])),
        "mean_delta_p_target": _mean(lists.get("delta_target", [])),
        "trajectory_n": len(lists.get("delta_correct", [])),
        **{key: value for role in ("honest", "confederate")
           for group in ("honest", "confederate") for key, value in peer_block(role, group).items()},
        **group_block("honest"), **group_block("confederate"),
    }
    return row


def summarize(trials: Iterable[Dict]) -> List[Dict]:
    """One row per condition, per scenario and pooled over scenarios ("ALL")."""
    groups: Dict[Tuple, Dict[str, Dict]] = {}
    for trial in trials:
        for condition in trial["analysis_conditions"]:
            key = tuple(condition[field] for field in GROUP_FIELDS)
            groups.setdefault(key, {})[trial["trial_id"]] = trial
            pooled = (condition["experiment"], "ALL") + key[2:]
            groups.setdefault(pooled, {})[trial["trial_id"]] = trial
    rows = []
    for key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        members = groups[key]
        rows.append({**dict(zip(GROUP_FIELDS, key)),
                     "source_trial_ids": ";".join(sorted(members)),
                     **summarize_group(list(members.values()))})
    return rows


def _difference(left: Optional[float], right: Optional[float]) -> Optional[float]:
    return None if left is None or right is None else left - right


def contrasts(rows: Sequence[Dict]) -> List[Dict]:
    """Matched condition differences for both experiments."""
    index = {tuple(row[field] for field in GROUP_FIELDS): row for row in rows}
    output: List[Dict] = []
    metrics = ("answer_change_rate", "answer_switch_to_target_rate", "mean_delta_p_correct",
               "mean_delta_p_target", "final_vote_accuracy")

    def lookup(row: Dict, **changes) -> Optional[Dict]:
        condition = {**row, **changes}
        return index.get(tuple(condition[field] for field in GROUP_FIELDS))

    def difference(row: Dict, other: Optional[Dict], label: str) -> None:
        if other is not None:
            output.append({**{field: row[field] for field in GROUP_FIELDS}, "contrast": label,
                           "control_arm": other["arm"],
                           "control_order_condition": other["order_condition"],
                           "left_source_trial_ids": row["source_trial_ids"],
                           "right_source_trial_ids": other["source_trial_ids"],
                           **{f"diff_{metric}": _difference(row[metric], other[metric])
                              for metric in metrics}})

    for row in rows:
        if row["arm"] == "deceiver":
            for control in ("honest_advocate", "no_confederate"):
                order = "none" if control == "no_confederate" else row["order_condition"]
                difference(row, lookup(row, arm=control, order_condition=order),
                           f"deceiver - {control}")
        if row["order_condition"] == "random_nonadjacent":
            difference(row, lookup(row, order_condition="adjacent_first"),
                       "random_nonadjacent - adjacent_first")
        if row["experiment"] == "hidden_profile" and row["info_mode"] == "hidden":
            full, reveal = lookup(row, info_mode="full"), lookup(row, info_mode="reveal_all")
            output.append({
                **{field: row[field] for field in GROUP_FIELDS},
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
    """Every initial, discussion, and final report, once per physical source trial."""
    rows = []
    for trial in trials:
        roles = {person["name"]: person["kind"] for person in trial["participants"]}
        for order, entry in enumerate(trial["decisions"]):
            belief = entry["probabilities"]
            rows.append({
                "trial_id": trial["trial_id"], **{field: trial[field] for field in GROUP_FIELDS},
                "analysis_condition_ids": ";".join(condition["trial_id"]
                                                    for condition in trial["analysis_conditions"]),
                "agent": entry["agent"], "role": roles[entry["agent"]], "slot": entry["slot"],
                "stage": entry["stage"], "round": entry["round"], "decision_order": order,
                "view_length": entry["view_length"], "answer": entry["answer"],
                "valid": belief is not None,
                **{f"p_{letter}": (belief[letter] if belief else None) for letter in LETTERS},
                "p_correct": belief[trial["correct"]] if belief else None,
                "p_target": belief[trial["target"]] if belief else None,
                "private_top": unique_top(belief),
                "correct": trial["correct"], "target": trial["target"],
            })
    return rows
