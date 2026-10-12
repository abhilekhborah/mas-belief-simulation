import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mas_sim import metrics, prompts  # noqa: E402
from mas_sim.engine import EventSink, TrialRunner, TrialSpec, plurality, unique_top  # noqa: E402
from mas_sim.experiments import build_trials, planned_calls  # noqa: E402
from mas_sim.llm import FakeClient  # noqa: E402
from mas_sim.runner import run  # noqa: E402
from mas_sim.scenarios import CORRECT, SCENARIOS, knowledge_from, letter_map, verify_scenario  # noqa: E402

SETTINGS = {"probability_tolerance": 1e-6, "max_parse_retries": 2, "collect_forecasts": True}


class ScenarioTests(unittest.TestCase):
    def test_every_scenario_passes_hidden_profile_checks(self):
        for key, scenario in SCENARIOS.items():
            self.assertEqual(verify_scenario(scenario), [], key)

    def test_ground_truth_is_computed_not_hardcoded(self):
        for scenario in SCENARIOS.values():
            decision = scenario.decide(knowledge_from(scenario.all_facts()))
            self.assertEqual(decision.selected, CORRECT)
            self.assertEqual(sum(1 for reasons in decision.failed_controls.values() if not reasons), 1)

    def test_rotation_places_correct_answer_on_every_letter(self):
        self.assertEqual({letter_map(rotation)[CORRECT] for rotation in range(4)}, set("ABCD"))

    def test_prompts_do_not_leak_evaluator_terms(self):
        scenario = SCENARIOS["payment_approval"]
        text = prompts.render_system_prompt(
            prompts.render_description(scenario, 0, "Person 3", 6, 3), scenario.all_facts())
        for word in ("decoy", "ground truth", "script", "correct answer", "hidden", "minority"):
            self.assertNotIn(word, text.lower())


class MetricTests(unittest.TestCase):
    def test_brier_and_plurality(self):
        self.assertAlmostEqual(metrics.brier({"A": .1, "B": .6, "C": .25, "D": .05}, "B", "ABCD"), .235)
        self.assertEqual(plurality(["B", "B", "C"]), "B")
        self.assertEqual(plurality(["B", "C"]), "Tie")
        self.assertEqual(plurality(["B", None]), "Incomplete")
        self.assertIsNone(unique_top({"A": .5, "B": .5, "C": 0, "D": 0}))

    def test_conversion_and_compliance_counting(self):
        trial = {
            "correct": "C", "target": "B", "group_outcome": "B", "model_group_outcome": "B",
            "participants": [{"name": "Person 1", "kind": "script"}, {"name": "Person 2", "kind": "model"}],
            "initial_votes": {"Person 2": "C"}, "final_votes": {"Person 1": "B", "Person 2": "B"},
            "probes": [
                {"agent": "Person 2", "stage": "baseline", "belief": {"A": 0, "B": .1, "C": .9, "D": 0},
                 "peer_forecasts": {"Person 1": {"A": 0, "B": 1, "C": 0, "D": 0}}, "forecast_round": 1,
                 "last_observed": {"Person 1": None}, "group_forecast": None},
                {"agent": "Person 2", "stage": "terminal", "belief": {"A": 0, "B": .7, "C": .3, "D": 0},
                 "peer_forecasts": {}, "forecast_round": None, "last_observed": {}, "group_forecast": None},
            ],
            "public": [
                {"round": 1, "speaker": "Person 1", "kind": "script", "answer": "B"},
                {"round": 1, "speaker": "Person 2", "kind": "model", "answer": "B", "private_top_at_turn": "C"},
            ],
        }
        result = metrics.trial_metrics(trial)
        self.assertEqual((result["converted"], result["conversion_eligible"]), (1, 1))
        self.assertEqual((result["complied"], result["compliance_eligible"]), (1, 1))
        self.assertAlmostEqual(result["delta_correct"][0], -0.6)
        self.assertEqual(result["peer_script"], [0.0])


class PipelineTests(unittest.TestCase):
    def test_hidden_trial_gives_each_member_one_unique_fact(self):
        spec = TrialSpec("hidden_profile", "privacy_report_release", 1, 0, 4, 0, "none", "hidden", 2, 7)
        record = TrialRunner(spec, FakeClient(seed=1), SETTINGS, EventSink(None)).run()
        unique = [fact for person in record["participants"] for fact in person["unique_facts"]]
        self.assertEqual(sorted(unique), ["prv_k1", "prv_k2", "prv_k3", "prv_k4"])
        self.assertEqual(record["model_calls"] >= planned_calls(spec), True)

    def test_full_offline_run_writes_all_outputs(self):
        config_path = os.path.join(ROOT, "config.json")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        config["llm"]["fake_invalid_rate"] = 0.05
        config["run"]["parallel_trials"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            run_dir = run(path, fake=True, output_dir=directory)
            for name in ("summary.csv", "contrasts.csv", "trajectories.csv", "summary.json",
                         "trials.jsonl", "calls.jsonl", "run_config.json"):
                self.assertTrue(os.path.exists(os.path.join(run_dir, name)), name)
            with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["failed_trials"], [])
            self.assertEqual(sum(row["trials"] for row in summary["summary"] if row["scenario"] != "ALL"),
                             len(build_trials(config)))
            with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as handle:
                self.assertNotIn("api_key", handle.read())


if __name__ == "__main__":
    unittest.main()
