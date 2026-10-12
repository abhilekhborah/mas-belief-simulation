"""Runs one trial: baseline -> initial vote -> R rounds -> terminal -> final vote.

Per trial, for every ordinary model member:
  1. Baseline private probe (no discussion), with forecasts of round-1 answers.
  2. Initial vote (HiddenBench style), recorded but not shown to anyone.
  3. Rounds 1..R in fixed order (scripted speakers first). At each model turn
     the visible transcript is frozen; a private probe and a separate public
     turn are made from that same view. Only the public answer and message
     are appended to the transcript.
  4. Terminal private probe with the complete transcript.
  5. Final vote with the complete transcript. Final votes are independent and
     are not shown to other members.
Scripted speakers have no prompts, no private state, and cast their scripted
option as their final vote.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import prompts
from .llm import ModelError, parse_json_object
from .scenarios import CORRECT, DECOY, LETTERS, SCENARIOS, TARGET, knowledge_from, letter_map

ARMS_WITH_SCRIPTS = ("consistent_wrong", "truthful_sham")


@dataclass(frozen=True)
class TrialSpec:
    experiment: str
    scenario: str
    item: int
    repeat: int
    n_models: int
    n_scripts: int
    arm: str
    info_mode: str
    rounds: int
    base_seed: int

    @property
    def composition(self) -> str:
        return f"{self.n_models}+{self.n_scripts}"

    @property
    def active_scripts(self) -> int:
        return self.n_scripts if self.arm in ARMS_WITH_SCRIPTS else 0

    @property
    def rotation(self) -> int:
        return (list(SCENARIOS).index(self.scenario) + self.item) % len(LETTERS)

    @property
    def trial_id(self) -> str:
        return (f"{self.experiment}/{self.scenario}/item{self.item}/rep{self.repeat}/"
                f"{self.composition}/{self.arm}/{self.info_mode}/R{self.rounds}")

    def to_dict(self) -> Dict:
        return {
            "trial_id": self.trial_id, "experiment": self.experiment, "scenario": self.scenario,
            "item": self.item, "repeat": self.repeat, "composition": self.composition,
            "n_models": self.n_models, "n_scripts_configured": self.n_scripts,
            "n_scripts_active": self.active_scripts, "arm": self.arm, "info_mode": self.info_mode,
            "rounds": self.rounds, "rotation": self.rotation,
        }


class EventSink:
    """Thread-safe JSONL writer for every model call attempt."""

    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self.lock = threading.Lock()

    def write(self, record: Dict) -> None:
        if self.path is None:
            return
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def validate_distribution(value, keys: Sequence[str], tolerance: float) -> Optional[str]:
    if not isinstance(value, dict):
        return "not an object"
    missing = [key for key in keys if key not in value]
    if missing:
        return f"missing keys {missing}"
    numbers = []
    for key in keys:
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return f"{key} is not a number"
        if not 0.0 <= float(number) <= 1.0:
            return f"{key} is outside [0, 1]"
        numbers.append(float(number))
    if abs(sum(numbers) - 1.0) > tolerance:
        return f"probabilities sum to {sum(numbers):.6f}, not 1"
    return None


def _clean(value: Dict, keys: Sequence[str]) -> Dict[str, float]:
    return {key: float(value[key]) for key in keys}


class TrialRunner:
    def __init__(self, spec: TrialSpec, client, settings: Dict, sink: EventSink) -> None:
        self.spec = spec
        self.client = client
        self.sink = sink
        self.tolerance = float(settings.get("probability_tolerance", 1e-6))
        self.max_parse_retries = int(settings.get("max_parse_retries", 2))
        self.collect_forecasts = bool(settings.get("collect_forecasts", True))
        self.calls = 0
        self.invalid_calls = 0
        self.usage: Dict[str, int] = {}

    # -- model call with validation and correction retries --------------------
    def _call(self, agent: str, system_prompt: str, user_prompt: str, spec: Dict,
              validator) -> Tuple[Optional[Dict], List[str]]:
        messages = prompts.build_messages(system_prompt, user_prompt)
        errors: List[str] = []
        parsed: Optional[Dict] = None
        for attempt in range(self.max_parse_retries + 1):
            self.calls += 1
            try:
                completion = self.client.complete(messages, spec)
                text = completion.text
                for key, value in completion.usage.items():
                    self.usage[key] = self.usage.get(key, 0) + value
                latency = completion.latency_seconds
            except ModelError as error:
                text, latency = "", 0.0
                errors = [f"model error: {error}"]
                self._log(agent, spec, attempt, messages, text, errors, latency)
                break
            obj = parse_json_object(text)
            errors = ["reply is not a JSON object"] if obj is None else validator(obj)
            self._log(agent, spec, attempt, messages, text, errors, latency)
            if obj is not None:
                parsed = obj
            if not errors:
                return obj, []
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Your reply was invalid: " + "; ".join(errors)
                 + ". Return only valid JSON in the requested shape."},
            ]
        self.invalid_calls += 1
        return parsed, errors

    def _log(self, agent, spec, attempt, messages, text, errors, latency) -> None:
        self.sink.write({
            "trial_id": self.spec.trial_id, "agent": agent, "call_kind": spec["call_kind"],
            "stage": spec.get("stage"), "round": spec.get("round"), "attempt": attempt,
            "messages": messages, "response": text, "valid": not errors, "errors": errors,
            "latency_seconds": round(latency, 3),
        })

    # -- call types ------------------------------------------------------------
    def probe(self, agent: str, system_prompt: str, view: List[Dict], stage: str,
              round_number: int, forecast_round: Optional[int], others: Sequence[str],
              slot: int) -> Dict:
        peers = list(others) if (self.collect_forecasts and forecast_round is not None) else []
        ask_group = self.collect_forecasts
        user_prompt = prompts.private_probe_prompt(
            prompts.render_transcript(view, agent), peers, forecast_round if peers else None, ask_group)

        def validator(obj: Dict) -> List[str]:
            problems = []
            error = validate_distribution(obj.get("private_belief"), LETTERS, self.tolerance)
            if error:
                problems.append(f"private_belief: {error}")
            if peers:
                forecasts = obj.get("peer_forecasts")
                if not isinstance(forecasts, dict):
                    problems.append("peer_forecasts: not an object")
                else:
                    for peer in peers:
                        error = validate_distribution(forecasts.get(peer), LETTERS, self.tolerance)
                        if error:
                            problems.append(f"peer_forecasts[{peer}]: {error}")
            if ask_group:
                error = validate_distribution(obj.get("group_forecast"), prompts.GROUP_OUTCOMES,
                                              self.tolerance)
                if error:
                    problems.append(f"group_forecast: {error}")
            return problems

        spec = {"call_kind": "private_probe", "stage": stage, "round": round_number,
                "peers": peers, "ask_group": ask_group}
        obj, errors = self._call(agent, system_prompt, user_prompt, spec, validator)
        obj = obj or {}
        belief = obj.get("private_belief")
        belief = _clean(belief, LETTERS) if validate_distribution(belief, LETTERS, self.tolerance) is None else None
        peer_forecasts = {}
        raw_peers = obj.get("peer_forecasts") if isinstance(obj.get("peer_forecasts"), dict) else {}
        for peer in peers:
            if validate_distribution(raw_peers.get(peer), LETTERS, self.tolerance) is None:
                peer_forecasts[peer] = _clean(raw_peers[peer], LETTERS)
        group = obj.get("group_forecast")
        group = (_clean(group, prompts.GROUP_OUTCOMES)
                 if ask_group and validate_distribution(group, prompts.GROUP_OUTCOMES, self.tolerance) is None
                 else None)
        last_observed = {}
        for entry in view:
            if entry["kind"] != "disclosure" and entry["answer"] is not None:
                last_observed[entry["speaker"]] = entry["answer"]
        return {
            "agent": agent, "stage": stage, "round": round_number, "slot": slot,
            "view_length": len(view), "belief": belief, "peer_forecasts": peer_forecasts,
            "forecast_round": forecast_round if peers else None, "group_forecast": group,
            "last_observed": {peer: last_observed.get(peer) for peer in peers},
            "errors": errors,
        }

    def public_turn(self, agent: str, system_prompt: str, view: List[Dict],
                    round_number: int) -> Tuple[Optional[str], str, List[str]]:
        user_prompt = prompts.public_turn_prompt(view, prompts.render_transcript(view, agent),
                                                 round_number, self.spec.rounds)

        def validator(obj: Dict) -> List[str]:
            problems = []
            if obj.get("public_answer") not in LETTERS:
                problems.append("public_answer must be one of A, B, C, D")
            message = obj.get("public_message")
            if not isinstance(message, str) or not message.strip():
                problems.append("public_message must be a non-empty string")
            return problems

        spec = {"call_kind": "public_turn", "stage": "public_turn", "round": round_number}
        obj, errors = self._call(agent, system_prompt, user_prompt, spec, validator)
        if errors:
            return None, "", errors
        return obj["public_answer"], obj["public_message"].strip(), []

    def vote(self, agent: str, system_prompt: str, transcript: Optional[str],
             stage: str) -> Tuple[Optional[str], str, List[str]]:
        def validator(obj: Dict) -> List[str]:
            problems = []
            if obj.get("vote") not in LETTERS:
                problems.append("vote must be one of A, B, C, D")
            if not isinstance(obj.get("rationale"), str):
                problems.append("rationale must be a string")
            return problems

        spec = {"call_kind": stage, "stage": stage, "round": None}
        obj, errors = self._call(agent, system_prompt, prompts.vote_prompt(transcript), spec, validator)
        if errors:
            return None, "", errors
        return obj["vote"], obj["rationale"], []

    # -- the trial ---------------------------------------------------------------
    def run(self) -> Dict:
        spec = self.spec
        scenario = SCENARIOS[spec.scenario]
        letters = letter_map(spec.rotation)
        correct, target, decoy = letters[CORRECT], letters[TARGET], letters[DECOY]
        n_scripts = spec.active_scripts
        names = [f"Person {number}" for number in range(1, spec.n_models + n_scripts + 1)]
        script_names, model_names = names[:n_scripts], names[n_scripts:]
        script_answer = target if spec.arm == "consistent_wrong" else correct

        packets: Dict[str, List] = {}
        unique_facts: Dict[str, List] = {}
        for slot, name in enumerate(model_names):
            if spec.info_mode == "full":
                critical = list(scenario.critical_facts)
            else:
                critical = [fact for index, fact in enumerate(scenario.critical_facts)
                            if (index + spec.item) % spec.n_models == slot]
            unique_facts[name] = critical
            facts = list(scenario.shared_facts) + critical
            random.Random(f"{spec.base_seed}|{spec.scenario}|{spec.item}|{spec.repeat}|{slot}").shuffle(facts)
            packets[name] = facts

        systems = {
            name: prompts.render_system_prompt(
                prompts.render_description(scenario, spec.rotation, name, len(names), spec.rounds),
                packets[name])
            for name in model_names
        }

        def others(name: str) -> List[str]:
            return [other for other in names if other != name]

        probes: List[Dict] = []
        transcript: List[Dict] = []
        initial_votes: Dict[str, Optional[str]] = {}

        for name in model_names:
            probes.append(self.probe(name, systems[name], [], "baseline", 0, 1, others(name),
                                     names.index(name)))
        for name in model_names:
            initial_votes[name], _, _ = self.vote(name, systems[name], None, "initial_vote")

        if spec.info_mode == "reveal_all":
            # Adapted from HiddenBench's Reveal-All ablation: each member's
            # unique information is posted to the transcript before round 1.
            for name in model_names:
                for fact in unique_facts[name]:
                    transcript.append({"round": 0, "speaker": name, "kind": "disclosure",
                                       "answer": None, "message": fact.text})

        for round_number in range(1, spec.rounds + 1):
            next_round = round_number + 1 if round_number < spec.rounds else None
            for name in names:
                if name in script_names:
                    transcript.append({"round": round_number, "speaker": name, "kind": "script",
                                       "answer": script_answer,
                                       "message": prompts.script_message(script_answer)})
                    continue
                view = list(transcript)
                probe = self.probe(name, systems[name], view, "pre_turn", round_number, next_round,
                                   others(name), names.index(name))
                probes.append(probe)
                answer, message, errors = self.public_turn(name, systems[name], view, round_number)
                if answer is None:
                    continue
                transcript.append({"round": round_number, "speaker": name, "kind": "model",
                                   "answer": answer, "message": message,
                                   "private_top_at_turn": unique_top(probe["belief"])})

        for name in model_names:
            probes.append(self.probe(name, systems[name], list(transcript), "terminal", spec.rounds,
                                     None, others(name), names.index(name)))
        final_votes: Dict[str, Optional[str]] = {name: script_answer for name in script_names}
        for name in model_names:
            final_votes[name], _, _ = self.vote(
                name, systems[name], prompts.render_transcript(transcript, name), "final_vote")

        decision = scenario.decide(knowledge_from(scenario.all_facts()))
        return {
            **spec.to_dict(),
            "letters": {letters[index]: scenario.option_names[index] for index in range(4)},
            "correct": correct, "target": target, "decoy": decoy,
            "script_answer": script_answer if script_names else None,
            "ground_truth_audit": {letters[index]: list(reasons)
                                   for index, reasons in decision.failed_controls.items()},
            "mit_risk_primary": scenario.mit_risk_primary,
            "mit_risk_hidden_mechanism": scenario.mit_risk_hidden_mechanism,
            "participants": [{"name": name, "kind": "script" if name in script_names else "model",
                              "unique_facts": [fact.fact_id for fact in unique_facts.get(name, [])]}
                             for name in names],
            "probes": probes,
            "public": [entry for entry in transcript if entry["kind"] != "disclosure"],
            "disclosures": [entry for entry in transcript if entry["kind"] == "disclosure"],
            "initial_votes": initial_votes,
            "final_votes": final_votes,
            "group_outcome": plurality(list(final_votes.values())),
            "model_group_outcome": plurality([final_votes[name] for name in model_names]),
            "model_calls": self.calls,
            "invalid_calls": self.invalid_calls,
            "usage": self.usage,
        }


def unique_top(belief: Optional[Dict[str, float]], tolerance: float = 1e-6) -> Optional[str]:
    """Unique highest-probability option; None for invalid or tied reports."""
    if not belief:
        return None
    best = max(belief.values())
    leaders = [key for key, value in belief.items() if best - value <= tolerance]
    return leaders[0] if len(leaders) == 1 else None


def plurality(ballots: Sequence[Optional[str]]) -> str:
    if not ballots or any(ballot is None for ballot in ballots):
        return "Incomplete"
    counts: Dict[str, int] = {}
    for ballot in ballots:
        counts[ballot] = counts.get(ballot, 0) + 1
    best = max(counts.values())
    leaders = [option for option, count in counts.items() if count == best]
    return leaders[0] if len(leaders) == 1 else "Tie"
