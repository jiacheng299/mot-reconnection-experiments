import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from hashlib import sha256

from reconnect_study.supplement_matched_eval import evaluate_payload, report_tables, validate_lock_files


def row(source, *, accepted, wrong, unverified, ignored, correct, gaps=10):
    return {"name": source, "source_id": source, "frames": 10, "split": "test",
            "accepted": accepted, "wrong": wrong, "unverified_accepted": unverified,
            "ignored_accepted": ignored, "correct": correct, "gaps": gaps}


class MatchedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "tracker": "botsort-motion-only", "split": "test", "acceptance_cap": 6,
            "policies": {
                "assignment_0": [row("a", accepted=4, wrong=2, unverified=1, ignored=2, correct=3),
                                 row("b", accepted=2, wrong=0, unverified=0, ignored=1, correct=2)],
                "linear_0": [row("a", accepted=3, wrong=1, unverified=1, ignored=3, correct=4),
                             row("b", accepted=3, wrong=1, unverified=0, ignored=0, correct=2)],
                "structured_0": [row("a", accepted=3, wrong=1, unverified=1, ignored=3, correct=4),
                                 row("b", accepted=3, wrong=0, unverified=0, ignored=0, correct=3)],
            },
            "overall": {
                "assignment_0": {"aggregate": {"HOTA": 40.0, "AssA": 30.0, "IDF1": 35.0, "IDSW": 8}},
                "linear_0": {"aggregate": {"HOTA": 41.0, "AssA": 31.0, "IDF1": 36.0, "IDSW": 7}},
                "structured_0": {"aggregate": {"HOTA": 42.0, "AssA": 32.0, "IDF1": 37.0, "IDSW": 6}},
            },
        }
        self.lock = {"acceptance_cap": 6, "test_policies": list(self.payload["policies"]),
                     "selection": {"points": {"mid": {
                         "target": 3.0, "tolerance": 0.15, "spread": 0.0, "matched": True,
                         "nearest": {
                             "assignment": {"policy": "assignment_0", "threshold": 0.2,
                                            "mean_accepted": 3.0, "deviation": 0.0},
                             "linear": {"policy": "linear_0", "threshold": 0.3,
                                        "mean_accepted": 3.0, "deviation": 0.0},
                             "structured": {"policy": "structured_0", "threshold": 0.4,
                                            "mean_accepted": 3.0, "deviation": 0.0},
                         }}}}}

    def test_summarizes_error_subset_distribution_cap_and_hota(self):
        report = evaluate_payload(self.payload, self.lock, replicas=200, seed=362)
        summary = report["policies"]["assignment_0"]
        self.assertEqual(summary["accepted"], 6)
        self.assertEqual(summary["wrong"], 2)
        self.assertEqual(summary["unverified_accepted"], 1)
        self.assertEqual(summary["unverified_is_wrong_subset"], True)
        self.assertEqual(summary["gate_actions"], 9)
        self.assertEqual(summary["sources_reaching_cap"], 1)
        self.assertEqual(summary["accepted_per_source"]["quartiles"], [2.5, 3.0, 3.5])
        self.assertEqual(summary["overall"]["HOTA"], 40.0)

    def test_compares_structured_minus_each_reference_with_paired_ci(self):
        report = evaluate_payload(self.payload, self.lock, replicas=200, seed=362)
        comparisons = report["matched_sets"][0]["comparisons"]
        self.assertEqual(set(comparisons), {"assignment", "linear"})
        self.assertAlmostEqual(
            comparisons["assignment"]["effects"]["wrong_per_acceptance"]["delta_percentage_points"],
            100 * (1 / 6 - 2 / 6),
        )
        self.assertAlmostEqual(
            comparisons["linear"]["effects"]["gap_recovery"]["delta_percentage_points"], 5.0
        )
        self.assertEqual(comparisons["assignment"]["resampling_replicas"], 200)

    def test_rejects_unverified_count_outside_wrong(self):
        self.payload["policies"]["assignment_0"][0]["unverified_accepted"] = 3
        with self.assertRaisesRegex(ValueError, "subset of wrong"):
            evaluate_payload(self.payload, self.lock, replicas=10)

    def test_exports_policy_and_comparison_rows(self):
        report = evaluate_payload(self.payload, self.lock, replicas=20, seed=362)
        policies, comparisons = report_tables(report)
        self.assertEqual(len(policies), 3)
        self.assertEqual(len(comparisons), 4)
        self.assertIn("correct_accepted", policies[0])
        self.assertIn("test_within_frozen_tolerance", comparisons[0])

    def test_unmatched_calibration_point_is_retained_without_test_comparison(self):
        self.lock["selection"]["points"]["mid"]["matched"] = False
        self.lock["test_policies"] = []
        report = evaluate_payload(self.payload, self.lock, replicas=20)
        point = report["matched_sets"][0]
        self.assertEqual(point["evaluation_status"], "not_run_unmatched_at_calibration")
        self.assertEqual(point["comparisons"], {})
        self.assertIsNone(point["spread_only_within_tolerance"])

    def test_rejects_lock_or_payload_cap_mismatch(self):
        self.lock["acceptance_cap"] = 12
        with self.assertRaisesRegex(ValueError, "acceptance cap"):
            evaluate_payload(self.payload, self.lock, replicas=10)

    def test_unconstrained_policy_has_no_cap_trigger_interpretation(self):
        self.payload["policies"]["unconstrained_tracker"] = [
            row("a", accepted=9, wrong=2, unverified=1, ignored=0, correct=4)
        ]
        self.payload["overall"]["unconstrained_tracker"] = {"aggregate": {"HOTA": 39.0}}
        self.lock["test_policies"].append("unconstrained_tracker")
        report = evaluate_payload(self.payload, self.lock, replicas=10)
        summary = report["policies"]["unconstrained_tracker"]
        self.assertIsNone(summary["sources_reaching_cap"])
        self.assertEqual(summary["cap_interpretation"], "unconstrained; no cap applies")

    def test_validates_config_calibration_and_inputs_bound_by_lock(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cap = root / "bot" / "B6"
            cap.mkdir(parents=True)
            config = cap / "config.json"
            calibration = cap / "calibration-dense.json"
            inputs = cap.parent / "input-splits.json"
            for path, value in ((config, {"acceptance_cap": 6}),
                                (calibration, {"split": "calibration"}),
                                (inputs, {"scope": "frozen"})):
                path.write_text(json.dumps(value) + "\n")
            lock = {"config_sha256": sha256(config.read_bytes()).hexdigest(),
                    "calibration_file": calibration.name,
                    "calibration_sha256": sha256(calibration.read_bytes()).hexdigest(),
                    "inputs_sha256": sha256(inputs.read_bytes()).hexdigest()}
            validate_lock_files(lock, cap / "locked.json", config)
            config.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "configuration"):
                validate_lock_files(lock, cap / "locked.json", config)


if __name__ == "__main__":
    unittest.main()
