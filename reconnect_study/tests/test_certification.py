import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from reconnect_study.certification import certify_manifest


def row(index, *, source_id=None, accepted=1, wrong=0, correct=0):
    return {"name": f"sequence-{index}", "source_id": source_id or f"source-{index}",
            "accepted": accepted, "wrong": wrong, "correct": correct, "gaps": 2}


def cohort(**overrides):
    return {"description": "Synthetic independent sources", "iid_source_sampling": True,
            "policies_frozen_before_calibration": True, **overrides}


class CertificationCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def write(self, name, data):
        path = self.path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def certify(self, manifest, min_active_coverage=0):
        return certify_manifest(self.write("manifest.json", manifest), alpha=0.10, delta=0.05,
                                min_active_coverage=min_active_coverage)

    def test_exploratory_cohort_emits_conditional_diagnostics_but_cannot_select(self):
        manifest = {"cohort": cohort(iid_source_sampling=False, caveats=["Public convenience sample"]),
                    "policies": {"p": [row(i) for i in range(60)]}}
        result = self.certify(manifest)
        method = result["methods"]["active_video"]
        self.assertEqual(method["delta"], 0.05)
        self.assertTrue(method["policies"]["p"]["passes_statistical_test"])
        self.assertFalse(method["policies"]["p"]["certified"])
        self.assertIsNone(method["selected_policy"])
        self.assertEqual(result["cohort"]["status"], "exploratory")
        self.assertIn("Public convenience sample", result["cohort"]["caveats"])

    def test_aggregates_clips_under_the_same_source_and_splits_delta(self):
        paths = [self.write(f"clip{i}.json", {"p": row(i, source_id=f"source-{i // 2}")}).name
                 for i in range(100)]
        cap = {"max_accepted": 2, "enforcement": "Two-action budget for every source in the replay protocol",
               "enforced_in_replay_and_deployment": True}
        result = self.certify({"cohort": cohort(), "replays": paths, "event_weighted_cap": cap})
        self.assertEqual(result["source_count"], 50)
        self.assertEqual(result["sequence_count"], 100)
        self.assertEqual(result["utility"]["p"]["accepted"], 100)
        self.assertEqual(result["event_weighted_cap"], cap)
        for method in ("event_weighted", "active_video"):
            self.assertAlmostEqual(result["methods"][method]["delta"], 0.025)
            self.assertTrue(result["methods"][method]["policies"]["p"]["certified"])

    def test_selects_correct_recovery_under_minimum_empirical_coverage(self):
        policies = {
            "lr": [row(i, accepted=int(i < 80), correct=int(i < 80)) for i in range(100)],
            "hr": [row(i, correct=int(i < 70)) for i in range(100)],
            "bad": [row(i, wrong=int(i < 25), correct=int(i < 99)) for i in range(100)],
        }
        result = self.certify({"cohort": cohort(), "policies": policies}, min_active_coverage=0.90)
        method = result["methods"]["active_video"]
        self.assertTrue(method["policies"]["lr"]["certified"])
        self.assertFalse(method["policies"]["lr"]["eligible_for_selection"])
        self.assertFalse(method["policies"]["bad"]["certified"])
        self.assertEqual(method["selected_policy"], "hr")
        self.assertAlmostEqual(result["utility"]["hr"]["correct_per_gap"], 70 / 200)

    def test_rejects_observed_or_violated_cap(self):
        manifest = {"cohort": cohort(), "policies": {"p": [row(1, accepted=3)]},
                    "event_weighted_cap": {"max_accepted": 1, "enforcement": "Replay cap"}}
        with self.assertRaisesRegex(ValueError, "enforced_in_replay_and_deployment"):
            self.certify(manifest)
        manifest["event_weighted_cap"]["enforced_in_replay_and_deployment"] = True
        with self.assertRaisesRegex(ValueError, "cap exceeded"):
            self.certify(manifest)

    def test_rejects_duplicate_sequence_or_changed_cohort_within_source(self):
        rows = [row(1, source_id="source"), row(2, source_id="source")]
        manifest = {"cohort": cohort(), "policies": {"x": rows, "y": rows[:1]}}
        with self.assertRaisesRegex(ValueError, "same sequence cohort"):
            self.certify(manifest)
        manifest["policies"] = {"x": [rows[0], rows[0]]}
        with self.assertRaisesRegex(ValueError, "duplicate sequence"):
            self.certify(manifest)

    def test_cli_loads_study_policy_rows_and_writes_result(self):
        self.write("evaluation-rows.json", {"split": "calibration", "acceptance_cap": 1,
                                          "policies": {"policy": [row(i) for i in range(40)]}})
        manifest_path = self.write("manifest.json", {"cohort": cohort(),
                                                    "policy_rows": "evaluation-rows.json"})
        output = self.path / "out" / "certification.json"
        run = subprocess.run([sys.executable, "-m", "reconnect_study.certification", "--input",
                              str(manifest_path), "--output", str(output), "--alpha", "0.1",
                              "--delta", "0.05", "--test", "betting"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["methods"]["active_video"]["selected_policy"], "policy")
        self.assertNotIn("event_weighted", result["methods"])
        self.assertEqual(result["test"], "betting")
        self.assertEqual(result["methods"]["active_video"]["policies"]["policy"]["test"], "betting")

    def test_study_event_weighted_excludes_unconstrained_baseline_explicitly(self):
        self.write("study.json", {"split": "calibration", "acceptance_cap": 1,
                                 "policies": {"policy": [row(0)], "unconstrained_tracker": [row(0)]}})
        cap = {"max_accepted": 1, "enforcement": "Study replay cap", "enforced_in_replay_and_deployment": True}
        manifest = {"cohort": cohort(), "policy_rows": "study.json", "event_weighted_cap": cap}
        with self.assertRaisesRegex(ValueError, "unconstrained_tracker"):
            self.certify(manifest)
        manifest["include_policies"] = ["policy"]
        result = self.certify(manifest)
        self.assertEqual(set(result["methods"]["event_weighted"]["policies"]), {"policy"})
        cap["max_accepted"] = 2
        with self.assertRaisesRegex(ValueError, "recorded replay cap"):
            self.certify(manifest)

    def finite_fixture(self, *, overlap=False, omit_clip=False, wrong_rank=False):
        pool = [f"source-{i:02}" for i in range(64)]
        sample = pool[:60]
        benchmark_rows = [{**row(i, source_id=source), "split": "calibration" if i < 59 else "test"}
                          for i, source in enumerate(pool)]
        benchmark_rows.append({**row(65, source_id=pool[0]), "split": "calibration"})
        development = pool[0] if overlap else "development"
        benchmark_rows.extend([
            {**row(66, source_id=development), "split": "development"},
            {**row(67, source_id="official-test"), "split": "test"},
        ])
        benchmark = self.write("benchmark.json", {"sequences": benchmark_rows})
        benchmark_hash = sha256(benchmark.read_bytes()).hexdigest()
        config = self.write("frozen.json", {"tracker": "bytetrack", "acceptance_cap": 2,
                            "study_manifest_sha256": benchmark_hash,
                            "development_sources": [development], "thresholds": {"margin": [0.3]}})
        config_hash = sha256(config.read_bytes()).hexdigest()
        design = self.write("design.json", {
            "draw_method": "uniform-combination-rank-v1", "random_rank_decimal": "1" if wrong_rank else "0",
            "population": pool, "calibration_sample": sample, "held_out_complement": pool[60:],
            "tracker": "bytetrack", "method": "fixed-betting-mixture", "risk_target": 0.1, "delta": 0.05,
            "event_delta_fraction": 0.5, "acceptance_cap": 2, "config_sha256": config_hash,
            "benchmark_manifest_sha256": benchmark_hash,
        })
        selected = [entry for entry in benchmark_rows if entry["split"] in ("calibration", "test")
                    and entry["source_id"] in sample and (not omit_clip or entry["name"] != "sequence-65")]
        replays = self.write("replay.json", {"tracker": "bytetrack", "split": "calibration",
                             "acceptance_cap": 2, "config_sha256": config_hash,
                             "policies": {"margin_0": selected}})
        return {"cohort": cohort(iid_source_sampling=False), "finite_population_design": design.name,
                "benchmark_manifest": benchmark.name, "frozen_config": config.name, "policy_rows": replays.name,
                "event_weighted_cap": {"max_accepted": 2, "enforcement": "A frozen replay policy cap",
                                       "enforced_in_replay_and_deployment": True}}

    def test_fixed_pool_design_validates_rank_population_family_and_replays(self):
        result = certify_manifest(self.write("manifest.json", self.finite_fixture()), alpha=0.1, delta=0.05,
                                  test="betting")
        self.assertEqual(result["cohort"]["status"], "finite_population_design_verified")
        self.assertEqual(result["finite_population_design"]["population_size"], 64)
        self.assertEqual(result["finite_population_design"]["held_out_size"], 4)
        self.assertEqual(result["source_count"], 60)
        self.assertEqual(result["sequence_count"], 61)
        certificate = result["methods"]["active_video"]["policies"]["margin_0"]
        self.assertTrue(certificate["certified"])

    def test_fixed_pool_rejects_overlap_missing_clips_or_nonreproducible_draw(self):
        for setting, error in (("overlap", "overlap"), ("omit_clip", "sampled source groups"),
                               ("wrong_rank", "random rank")):
            with self.subTest(setting=setting), self.assertRaisesRegex(ValueError, error):
                certify_manifest(self.write("manifest.json", self.finite_fixture(**{setting: True})),
                                 alpha=0.1, delta=0.05, test="betting")
