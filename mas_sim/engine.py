"""HiddenBench-style trial: initial vote, free-text rounds, a private vote after every round, forecasts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from . import prompts
from .llm import ModelError, parse_json_object
from .scenarios import CORRECT, DECOY, LETTERS, SCENARIOS, knowledge_from, letter_map

CONFEDERATE_ARMS = ("deceiver", "honest_advocate")


@dataclass(frozen=True)
class TrialSpec:
    experiment: str
    scenario: str
    item: int
    repeat: int
    n_honest: int
    n_confederates: int
    arm: str
    info_mode: str
    rounds: int
    base_seed: int
    order_condition: str = "adjacent_first"
    analysis_conditions: Tuple[Dict, ...] = ()

    @property
    def composition(self) -> str:
        return f"{self.n_honest}+{self.n_confederates}"

    @property
    def active_confederates(self) -> int:
        return self.n_confederates if self.arm in CONFEDERATE_ARMS else 0

    @property
    def lure_index(self) -> int:
        return DECOY

    @property
    def goal_index(self) -> int:
        return CORRECT if self.arm == "honest_advocate" else self.lure_index

    @property
    def rotation(self) -> int:
        return (list(SCENARIOS).index(self.scenario) + self.item) % len(LETTERS)

    @property
    def trial_id(self) -> str:
        return (f"{self.experiment}/{self.scenario}/item{self.item}/rep{self.repeat}/"
                f"{self.composition}/{self.arm}/{self.info_mode}/{self.order_condition}/R{self.rounds}")

    @property
    def order_seed(self) -> str:
        return (f"{self.base_seed}|{self.scenario}|{self.item}|{self.repeat}|"
                f"{self.n_honest}|{self.active_confederates}|speaking_order")

    def speaking_order(self, confederates: List[str], honest: List[str]) -> List[str]:
        if self.order_condition != "random_nonadjacent":
            return confederates + honest
        positions = [indices for indices in combinations(range(len(confederates) + len(honest)),
                                                         len(confederates))
                     if all(right - left > 1 for left, right in zip(indices, indices[1:]))]
        if len(confederates) < 2 or not positions:
            raise ValueError("random_nonadjacent needs at least two non-adjacent confederates")
        rng = random.Random(self.order_seed)
        selected = rng.choice(positions)
        shuffled = list(confederates)
        rng.shuffle(shuffled)
        confederate_iter, honest_iter = iter(shuffled), iter(honest)
        return [next(confederate_iter) if slot in selected else next(honest_iter)
                for slot in range(len(confederates) + len(honest))]

    def condition(self, rounds: Optional[int] = None) -> Dict:
        """Analysis condition; ``rounds`` below the trial's length reads the vote after that round."""
        rounds = self.rounds if rounds is None else rounds
        trial_id = self.trial_id.rsplit("/R", 1)[0] + f"/R{rounds}"
        return {"trial_id": trial_id, "experiment": self.experiment,
                "scenario": self.scenario, "composition": self.composition, "arm": self.arm,
                "info_mode": self.info_mode, "rounds": rounds,
                "order_condition": self.order_condition}

    def to_dict(self) -> Dict:
        return {
            "trial_id": self.trial_id, "experiment": self.experiment, "scenario": self.scenario,
            "item": self.item, "repeat": self.repeat, "composition": self.composition,
            "n_honest": self.n_honest, "n_confederates_configured": self.n_confederates,
            "n_confederates_active": self.active_confederates, "arm": self.arm,
            "info_mode": self.info_mode,
            "rounds": self.rounds, "rotation": self.rotation,
            "order_condition": self.order_condition,
            "analysis_conditions": list(self.analysis_conditions) or [self.condition()],
        }


def now_iso(precision: str = "milliseconds") -> str:
    """Local time with its UTC offset, e.g. 2026-10-14T01:31:14.250+05:30."""
    return datetime.now().astimezone().isoformat(timespec=precision)


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


def _normalize_option(text: str) -> str:
    return " ".join(text.lower().strip().rstrip(".").split())


class TrialRunner:
    def __init__(self, spec: TrialSpec, client, settings: Dict) -> None:
        self.spec = spec
        self.client = client
        self.call_log: List[Dict] = []
        self.tolerance = float(settings.get("probability_tolerance", 1e-6))
        self.max_parse_retries = int(settings.get("max_parse_retries", 2))
        self.collect_forecasts = bool(settings.get("collect_forecasts", True))
        self.calls = 0
        self.invalid_calls = 0
        self.usage: Dict[str, int] = {}
        self.option_labels: Dict[str, Tuple[str, ...]] = {}

    # -- model call with validation and correction retries --------------------
    def _call(self, agent: str, messages: List[Dict[str, str]], spec: Dict,
              validator=None) -> Tuple[Optional[object], List[str]]:
        """JSON reply checked by ``validator``, or free text when ``validator`` is None."""
        errors: List[str] = []
        parsed = None
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
            if validator is None:
                obj = text.strip() or None
                errors = [] if obj else ["reply is empty"]
            else:
                obj = parse_json_object(text)
                errors = ["reply is not a JSON object"] if obj is None else validator(obj)
            self._log(agent, spec, attempt, messages, text, errors, latency)
            if obj is not None:
                parsed = obj
            if not errors:
                return obj, []
            fix = ("Return only valid JSON in the requested shape." if validator
                   else "Write your message to the team.")
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Your reply was invalid: " + "; ".join(errors) + ". " + fix},
            ]
        self.invalid_calls += 1
        return parsed, errors

    def _log(self, agent, spec, attempt, messages, text, errors, latency) -> None:
        self.call_log.append({
            "call_number": len(self.call_log) + 1, "timestamp": now_iso(),
            "agent": agent, "call_kind": spec["call_kind"],
            "stage": spec.get("stage"), "round": spec.get("round"), "attempt": attempt,
            "view_event_ids": list(spec.get("view_event_ids", [])),
            "messages": messages, "response": text, "valid": not errors, "errors": errors,
            "latency_seconds": round(latency, 3),
        })

    # -- call types ------------------------------------------------------------
    def vote(self, name: str, conversation: List[Dict[str, str]], messages: Optional[List[str]],
             round_number: int, slot: int, view_ids: List[int]) -> Dict:
        """Private vote on a copy of the conversation; it never enters the agent's history."""
        def validator(obj: Dict) -> List[str]:
            problems = []
            if obj.get("vote") not in LETTERS:
                problems.append("vote must be one of A, B, C, D")
            error = validate_distribution(obj.get("probabilities"), LETTERS, self.tolerance)
            if error:
                problems.append(f"probabilities: {error}")
            if not isinstance(obj.get("rationale"), str) or not obj["rationale"].strip():
                problems.append("rationale must be a non-empty string")
            selected = obj.get("selected_option")
            if not isinstance(selected, str) or not selected.strip():
                problems.append("selected_option must be the text of the chosen option")
            elif obj.get("vote") in self.option_labels and _normalize_option(selected) not in {
                    _normalize_option(label) for label in self.option_labels[obj["vote"]]}:
                problems.append(f"vote {obj['vote']} and selected_option {selected!r} name different "
                                "options; they must refer to the same option")
            return problems

        stage = "initial" if round_number == 0 else "round"
        request = {"call_kind": "vote", "stage": stage, "round": round_number,
                   "view_event_ids": view_ids}
        obj, errors = self._call(
            name, conversation + [{"role": "user", "content": prompts.vote_prompt(messages)}],
            request, validator)
        valid = not errors and obj is not None
        probabilities = _clean(obj["probabilities"], LETTERS) if valid else None
        top = max(probabilities.values()) if probabilities else None
        return {"agent": name, "stage": stage, "round": round_number, "slot": slot,
                "view_length": len(view_ids), "view_event_ids": view_ids,
                "answer": obj["vote"] if valid else None,
                "selected_option": obj["selected_option"].strip() if valid else None,
                "probabilities": probabilities,
                # Expected to be False for confederates pushing a goal they do not believe.
                "answer_matches_probability_top": (
                    top - probabilities[obj["vote"]] <= self.tolerance) if valid else None,
                "rationale": obj["rationale"].strip() if valid else "", "errors": errors}

    def forecast(self, name: str, conversation: List[Dict[str, str]], messages: List[str],
                 peers: List[str], round_number: int, previous_votes: Dict[str, Optional[str]],
                 view_ids: List[int]) -> Dict:
        def validator(obj: Dict) -> List[str]:
            problems = []
            forecasts = obj.get("peer_forecasts")
            if not isinstance(forecasts, dict) or set(forecasts) != set(peers):
                problems.append("peer_forecasts must contain exactly the requested participants")
            else:
                for peer in peers:
                    error = validate_distribution(forecasts[peer], LETTERS, self.tolerance)
                    if error:
                        problems.append(f"peer_forecasts[{peer}]: {error}")
            error = validate_distribution(obj.get("group_forecast"), prompts.GROUP_OUTCOMES,
                                          self.tolerance)
            if error:
                problems.append(f"group_forecast: {error}")
            return problems

        request = {"call_kind": "forecast", "stage": "pre_round", "round": round_number,
                   "peers": peers, "view_event_ids": view_ids}
        obj, errors = self._call(
            name, conversation + [{"role": "user", "content": prompts.forecast_prompt(messages, peers)}],
            request, validator)
        valid = not errors and obj is not None
        return {"agent": name, "stage": "pre_round", "round": round_number,
                "forecast_round": round_number, "view_length": len(view_ids),
                "view_event_ids": view_ids,
                "peer_forecasts": {peer: _clean(obj["peer_forecasts"][peer], LETTERS)
                                   for peer in peers} if valid else {},
                "group_forecast": _clean(obj["group_forecast"], prompts.GROUP_OUTCOMES)
                                  if valid else None,
                "previous_vote": {peer: previous_votes.get(peer) for peer in peers}, "errors": errors}

    # -- the trial ---------------------------------------------------------------
    def run(self) -> Dict:
        started_at = now_iso()
        spec = self.spec
        scenario = SCENARIOS[spec.scenario]
        letters = letter_map(spec.rotation)
        correct, target, decoy = letters[CORRECT], letters[spec.lure_index], letters[DECOY]
        self.option_labels = {letters[index]: (scenario.option_phrases[index],
                                               scenario.option_names[index]) for index in range(4)}
        n_confederates = spec.active_confederates
        names = [f"Person {number}" for number in range(1, spec.n_honest + n_confederates + 1)]
        confederate_names, honest_names = names[:n_confederates], names[n_confederates:]
        order = spec.speaking_order(confederate_names, honest_names)
        goal = letters[spec.goal_index] if confederate_names else None

        packets: Dict[str, List] = {}
        unique_facts: Dict[str, List] = {}
        for slot, name in enumerate(honest_names):
            if spec.info_mode == "full":
                critical = list(scenario.critical_facts)
            else:
                critical = [fact for index, fact in enumerate(scenario.critical_facts)
                            if (index + spec.item) % spec.n_honest == slot]
            unique_facts[name] = critical
            facts = list(scenario.shared_facts) + critical
            random.Random(f"{spec.base_seed}|{spec.scenario}|{spec.item}|{spec.repeat}|{slot}").shuffle(facts)
            packets[name] = facts

        for name in confederate_names:
            facts = list(scenario.shared_facts) + list(scenario.critical_facts)
            random.Random(f"{spec.base_seed}|{spec.scenario}|{spec.item}|{spec.repeat}|{name}").shuffle(facts)
            packets[name] = facts

        cue = prompts.ASYMMETRY_CUE if spec.info_mode == "hidden_cue" else ""

        def extra(name: str) -> str:
            return cue if name in honest_names else ""

        def system_prompt(name: str) -> str:
            description = prompts.render_description(scenario, spec.rotation, name, len(names))
            if name not in confederate_names:
                return prompts.render_system_prompt(description, packets[name], extra(name))
            instruction = prompts.confederate_instruction(
                goal, scenario.option_phrases[spec.goal_index], spec.arm == "deceiver")
            return prompts.render_system_prompt(description, packets[name], instruction)

        conversations = {name: [{"role": "system", "content": system_prompt(name)}] for name in names}
        seen = {name: 0 for name in names}  # transcript events already in each conversation
        visible: Dict[str, List[int]] = {name: [] for name in names}
        decisions: List[Dict] = []
        forecasts: List[Dict] = []
        transcript: List[Dict] = []
        skipped_turns: List[Dict] = []
        round_votes: List[Dict[str, Optional[str]]] = []

        def lines(entries: List[Dict]) -> List[str]:
            return [f"{entry['speaker']}: {entry['message']}" for entry in entries]

        def shown(name: str, entries: List[Dict]) -> List[int]:
            return sorted(set(visible[name]) | {entry["event_id"] for entry in entries})

        def publish(entry: Dict) -> Dict:
            entry = {**entry, "event_id": len(transcript) + 1}
            transcript.append(entry)
            return entry

        for name in names:
            decisions.append(self.vote(name, conversations[name], None, 0, order.index(name), []))
        initial = {entry["agent"]: entry for entry in decisions}

        if spec.info_mode == "reveal_all":
            for name in honest_names:
                for fact in unique_facts[name]:
                    publish({"round": 0, "speaker": name, "kind": "disclosure", "message": fact.text})

        for round_number in range(1, spec.rounds + 1):
            if self.collect_forecasts and round_number >= 2:
                previous = [entry for entry in transcript if entry["round"] == round_number - 1]
                for name in names:
                    forecasts.append(self.forecast(
                        name, conversations[name], lines(previous),
                        [other for other in names if other != name], round_number, round_votes[-1],
                        shown(name, previous)))
            for name in order:
                new = [entry for entry in transcript[seen[name]:] if entry["speaker"] != name]
                first = initial[name] if round_number == 1 and initial[name]["answer"] else None
                user = {"role": "user", "content": prompts.turn_prompt(lines(new), first, extra(name))}
                request = {"call_kind": "turn", "stage": "discussion", "round": round_number,
                           "view_event_ids": shown(name, new)}
                text, errors = self._call(name, conversations[name] + [user], request)
                if text is None:
                    skipped_turns.append({"agent": name, "round": round_number, "errors": errors})
                    continue
                conversations[name] += [user, {"role": "assistant", "content": text}]
                entry = publish({"round": round_number, "speaker": name, "kind": "message",
                                 "message": text})
                visible[name] = shown(name, new + [entry])
                seen[name] = len(transcript)
            this_round = [entry for entry in transcript
                          if entry["round"] == round_number and entry["kind"] == "message"]
            votes: Dict[str, Optional[str]] = {}
            for name in names:
                result = self.vote(name, conversations[name], lines(this_round), round_number,
                                   order.index(name), shown(name, this_round))
                decisions.append(result)
                votes[name] = result["answer"]
            round_votes.append(votes)

        final = {entry["agent"]: entry for entry in decisions
                 if entry["stage"] == "round" and entry["round"] == spec.rounds}
        final_votes = round_votes[-1]

        decision = scenario.decide(knowledge_from(scenario.all_facts()))
        return {
            **spec.to_dict(),
            "started_at": started_at, "finished_at": now_iso(),
            "letters": {letters[index]: scenario.option_names[index] for index in range(4)},
            "correct": correct, "target": target, "decoy": decoy,
            "confederate_goal": goal,
            "speaking_order": order, "order_seed": spec.order_seed,
            "confederate_positions": [order.index(name) + 1 for name in confederate_names],
            "ground_truth_audit": {letters[index]: list(reasons)
                                   for index, reasons in decision.failed_controls.items()},
            "mit_risk_primary": scenario.mit_risk_primary,
            "mit_risk_hidden_mechanism": scenario.mit_risk_hidden_mechanism,
            "participants": [{"name": name, "kind": "confederate" if name in confederate_names
                                                       else "honest",
                              "unique_facts": [fact.fact_id for fact in unique_facts.get(name, [])]}
                             for name in names],
            "evidence": {name: [{"fact_id": fact.fact_id, "text": fact.text} for fact in packets[name]]
                         for name in names},
            "conversations": conversations,
            "decisions": decisions,
            "forecasts": forecasts,
            "public": [entry for entry in transcript if entry["kind"] != "disclosure"],
            "disclosures": [entry for entry in transcript if entry["kind"] == "disclosure"],
            "initial_votes": {name: initial[name]["answer"] for name in names},
            "initial_rationales": {name: initial[name]["rationale"] for name in names},
            "round_votes": round_votes,
            "final_votes": final_votes,
            "final_rationales": {name: final[name]["rationale"] for name in names},
            "skipped_turns": skipped_turns,
            "group_outcome": plurality(list(final_votes.values())),
            "honest_group_outcome": plurality([final_votes[name] for name in honest_names]),
            "model_calls": self.calls,
            "invalid_calls": self.invalid_calls,
            "usage": self.usage,
            "calls": self.call_log,
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
