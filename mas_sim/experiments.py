"""Experiment grid and the theory behind each experiment.

Experiment 1: minority conversion (complete information)
--------------------------------------------------------
Theory: consistent minority influence. Moscovici, Lage & Naffrechoux (1969),
"Influence of a consistent minority on the responses of a majority in a color
perception task", Sociometry 32(4). Four naive participants and two
confederates who consistently called blue slides "green"; consistency of the
minority produced a measurable shift toward the minority response compared
with an inconsistent minority and a control group.

Our adaptation: every model member receives all facts, so the correct option
is individually derivable. Scripted speakers (no prompt, no private state)
consistently recommend the minority target in every round
("consistent_wrong"). Matched controls keep everything else identical:
"truthful_sham" scripts recommend the correct option, and "no_script" removes
the scripts. Compositions (models + scripts): 4+2 (main, matching Moscovici's
4 + 2), 5+1, and 3+1. Every composition satisfies 0 < scripts < group/2.
Note: 4+2 vs 5+1 changes both minority size and model count, and 3+1 also
changes group size, so these are composition ablations rather than a clean
test of minority fraction.

Measured: reported conversion (private preference moves from correct to the
minority target), reported compliance (public answer differs from the
same-view private preference), probability trajectories, accuracy, and
forecasts. These are elicited reports, not access to internal beliefs.

Experiment 2: hidden-profile information pooling
------------------------------------------------
Theory: shared-information bias and hidden profiles. Stasser & Titus (1985),
"Pooling of unshared information in group decision making: Biased
information sampling during discussion", JPSP 48(6). Groups discussed shared
information more than unshared information and often missed the option that
the distributed (unshared) information supported.

Benchmark protocol: HiddenBench (Li, Naito & Shirado, arXiv:2505.11556).
Shared facts favour a decoy; each member receives one unique critical fact;
only pooling all of them reveals the correct option. Profiles:
  * full       every member receives every fact (individual upper bound),
  * hidden     shared facts + one unique critical fact per member,
  * reveal_all hidden packets, but every member's unique fact is posted to the
               transcript before round 1 (diagnostic: separates failure to
               share from failure to integrate).
Four model members, no scripts: a truthful script would hand over the answer
and invalidate the pooling test.

Both experiments run every scenario family at 1, 2 and 3 rounds as separate,
fresh trials (a 3-round trial is never truncated to make a 1-round result).

MIT AI Risk Repository classification (arXiv:2408.12622v3, Table 2): see
scenarios.py. Experiment 1 inherits each family's primary subdomain (4.3,
2.1, 7.3). Experiment 2 is additionally tagged 7.6 Multi-agent risks, whose
definition includes failures from "a lack of shared information" between
interacting agents.
"""

from __future__ import annotations

from typing import Dict, List

from .engine import ARMS_WITH_SCRIPTS, TrialSpec
from .scenarios import SCENARIOS

ALLOWED_ARMS = ARMS_WITH_SCRIPTS + ("no_script",)
ALLOWED_INFO_MODES = ("full", "hidden", "reveal_all")


def build_trials(config: Dict) -> List[TrialSpec]:
    run = config["run"]
    seed = int(run["seed"])
    items = int(run["items_per_scenario"])
    repeats = int(run["repeats"])
    scenario_keys = config["scenarios"]
    unknown = [key for key in scenario_keys if key not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Available: {list(SCENARIOS)}")

    trials: List[TrialSpec] = []
    minority = config.get("experiment_1_minority_conversion", {})
    if minority.get("enabled"):
        for arm in minority["arms"]:
            if arm not in ALLOWED_ARMS:
                raise ValueError(f"Unknown arm {arm}; use {ALLOWED_ARMS}")
        for n_models, n_scripts in minority["compositions"]:
            if not 0 < n_scripts < (n_models + n_scripts) / 2:
                raise ValueError(f"Composition {n_models}+{n_scripts} is not a minority")
            for scenario in scenario_keys:
                for item in range(items):
                    for repeat in range(repeats):
                        for arm in minority["arms"]:
                            for rounds in minority["rounds"]:
                                trials.append(TrialSpec(
                                    "minority_conversion", scenario, item, repeat, int(n_models),
                                    int(n_scripts), arm, "full", int(rounds), seed))

    hidden = config.get("experiment_2_hidden_profile", {})
    if hidden.get("enabled"):
        n_models = int(hidden["models"])
        for mode in hidden["info_modes"]:
            if mode not in ALLOWED_INFO_MODES:
                raise ValueError(f"Unknown info mode {mode}; use {ALLOWED_INFO_MODES}")
        for scenario in scenario_keys:
            needed = len(SCENARIOS[scenario].critical_facts)
            if n_models != needed:
                raise ValueError(f"{scenario} distributes {needed} critical facts; set models to {needed}")
            for item in range(items):
                for repeat in range(repeats):
                    for mode in hidden["info_modes"]:
                        for rounds in hidden["rounds"]:
                            trials.append(TrialSpec(
                                "hidden_profile", scenario, item, repeat, n_models, 0, "none", mode,
                                int(rounds), seed))
    return trials


def planned_calls(trial: TrialSpec) -> int:
    """Calls without retries: baseline + initial vote + 2 per round + terminal + final vote."""
    return trial.n_models * (2 * trial.rounds + 4)
