"""Run locked additive-15 policies and compare to audited Round 2 traces."""

import argparse
import csv
from pathlib import Path

from reconnect_study import study
from reconnect_study.data_io import digest
from reconnect_study.effect_sizes import compare
from reconnect_study.round3_ablation import Additive15, MANIFEST, OUTPUT, ROOT, compare_match
from reconnect_study.supplement_matched import load
from reconnect_study.supplement_matched_eval import summarize_policy


def bound(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Frozen input hash differs: {path}")


def config_path_in_checkout(path, *, source=False):
    portable = path.replace("\\", "/")
    if source:
        parts = portable.split("/")
        if parts[-2] != "reconnect_study" or not parts[-1].endswith(".py"):
            raise ValueError(f"Unexpected frozen source path: {path}")
        return Path(__file__).with_name(parts[-1])
    return Path(portable)


def verify_sources(result, expected):
    for policy, rows in result["policies"].items():
        actual = {(row["name"], row["source_id"], row["frames"]) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f"Unexpected test input roster for {policy}")
        if sum(row["gaps"] for row in rows) != 3155:
            raise ValueError(f"Unexpected natural-gap denominator for {policy}")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(tracker):
    directory = OUTPUT / tracker
    lock = load(directory / "locked.json")
    old = ROOT / "round2" / tracker / "B12"
    out = {
        "tracker": tracker, "acceptance_cap": 12, "lock_path": str(directory / "locked.json"),
        "lock_sha256": digest(directory / "locked.json"),
        "fit_path": str(directory / "fit.json"),
        "new_test_path": str(directory / "test.json"), "new_trace_root": str(directory / "traces"),
        "old_test_path": str(old / "test.json"), "old_trace_root": str(old / "traces"),
        "report_path": str(directory / "report.json"), "work_points": lock["work_points"],
    }
    study.save(directory / "manifest.json", out)


def run(key):
    directory = OUTPUT / key
    config_path = directory / "config.json"
    lock_path = directory / "locked.json"
    test_path = directory / "test.json"
    config, lock = load(config_path), load(lock_path)
    if test_path.exists():
        raise ValueError(f"Refusing to overwrite existing test replay: {test_path}")
    bound(config_path, lock["config_sha256"])
    bound(directory / lock["calibration_file"], lock["calibration_sha256"])
    calibration_archive = load(directory / lock["calibration_file"])
    bound(directory / "fit.json", calibration_archive["fit_sha256"])
    if calibration_archive["selection"] != lock["work_points"]:
        raise ValueError("Locked work points disagree with archived calibration")
    for item in config["inputs"].values():
        bound(config_path_in_checkout(item["path"]), item["sha256"])
    for path, sha256 in config["source_sha256"].items():
        bound(config_path_in_checkout(path, source=True), sha256)
    oldroot = ROOT / "round2" / key
    prior = oldroot / "B12"
    audited = load(oldroot / "test-integrity.json")["B12"]
    bound(prior / "test.json", audited["test_sha256"])
    archived = load(prior / "test.json")
    if (archived["benchmark_manifest_sha256"] != config["study_manifest_sha256"]
            or archived["config_sha256"] != config["inputs"]["old_config"]["sha256"]
            or archived["acceptance_cap"] != 12 or archived["tracker"] != config["tracker"]):
        raise ValueError("Prior online replay uses a different input, model or cap")
    expected = {(row["name"], row["source_id"], row["frames"])
                for row in load(MANIFEST)["sequences"] if row["split"] == "test"}
    verify_sources(archived, expected)
    if key == "byte":
        for name, failure, accepted in (("assignment_45", 158, 390), ("linear_40", 114, 393),
                                        ("structured_42", 98, 381)):
            rows = archived["policies"][name]
            if (sum(row["wrong"] for row in rows), sum(row["accepted"] for row in rows)) != (failure, accepted):
                raise ValueError("Expected ByteTrack B12 high anchor mismatch")
    write_manifest(key)
    new_model = Additive15(**config["score"])
    if lock["test_policies"]:
        old_specs = study.policy_specs
        study.policy_specs = lambda _config: {name: (new_model, cutoff)
                                               for name, cutoff in config["thresholds"].items()}
        try:
            study.run_split(MANIFEST, config_path, "test", test_path, lock["test_policies"],
                            "personpath22-native-keyframe", None, directory / "traces")
        finally:
            study.policy_specs = old_specs
        fresh = load(test_path)
        verify_sources(fresh, expected)
        for policy, rows in fresh["policies"].items():
            for row in rows:
                bound(directory / "traces" / policy / f"{row['name']}.json", row["trace_sha256"])
    else:
        fresh = {"policies": {}, "overall": {}}
        study.save(test_path, {**fresh, "status": "no_calibration_matched_policy"})
    data = {**archived["policies"], **fresh["policies"]}
    overall = {**archived["overall"], **fresh["overall"]}
    settings = lock["work_points"]
    descriptions = []
    effects = []
    point_reports = {}
    for point, setting in settings.items():
        target, tolerance = setting["target"], setting["tolerance"]
        roles = {**setting["old"], "summary15": setting["new"],
                 "unconstrained": {"policy": "unconstrained_tracker", "threshold": None}}
        metrics = {}
        for family, role in roles.items():
            name = role["policy"]
            row = {
                "tracker": config["tracker"], "cap": 12, "work_point": point, "family": family,
                "policy": name, "target": target, "threshold": role["threshold"], "gap_denominator": 3155,
                "scope": "Previously analyzed official PersonPath22 test split; exploratory sensitivity.",
            }
            if name in data:
                result = summarize_policy(data[name], cap=None if family == "unconstrained" else 12,
                                          overall=overall[name])
                metrics[family] = result
                row.update({field: result[field] for field in (
                    "accepted", "wrong", "unverified_accepted", "wrong_per_acceptance", "correct_accepted",
                    "active_sources", "ignored_accepted", "correct", "sources_reaching_cap",
                )})
                row["HOTA"] = result["overall"]["HOTA"]
            descriptions.append(row)
        paired = {}
        for label, changed, reference in (("23_minus_15", "structured", "summary15"),
                                          ("15_minus_12", "summary15", "linear")):
            calibration = setting["calibration_match"][label]
            result = {"calibration": calibration, "test": None}
            if changed in metrics and reference in metrics:
                a, b = metrics[changed], metrics[reference]
                mean_a = a["accepted_per_source"]["mean"]
                mean_b = b["accepted_per_source"]["mean"]
                test_match = compare_match(mean_a, mean_b, target, tolerance)
                effect = compare(data[roles[reference]["policy"]], data[roles[changed]["policy"]])
                result["test"] = {**test_match, "paired_effect": effect, "changed_accepted": a["accepted"],
                                  "reference_accepted": b["accepted"]}
                for metric, entry in effect["effects"].items():
                    lo, hi = entry["empirical_95_percent_interval"]
                    effects.append({
                        "tracker": config["tracker"], "cap": 12, "work_point": point, "comparison": label,
                        "changed_policy": roles[changed]["policy"], "reference_policy": roles[reference]["policy"],
                        "changed_accepted": a["accepted"], "reference_accepted": b["accepted"],
                        "target": target, "tolerance": tolerance, "calibration_matched": calibration["matched"],
                        "test_matched": test_match["matched"], "test_pairwise_matched": test_match["spread_within_tolerance"],
                        "test_absolute_target_matched": test_match["target_deviations_within_tolerance"],
                        "metric": metric, "delta_pp": entry["delta_percentage_points"], "ci95_low": lo,
                        "ci95_high": hi, "valid_replicas": entry["valid_replicas"],
                    })
            paired[label] = result
        point_reports[point] = {"policies": roles, "metrics": metrics, "comparisons": paired,
                                "target": target, "tolerance": tolerance}
    study.save(directory / "report.json", {
        "scope": "Previously analyzed official PersonPath22 test split; exploratory sensitivity.",
        "test_sha256": digest(test_path), "lock_sha256": digest(lock_path), "work_points": point_reports,
    })
    write_csv(directory / "results.csv", descriptions)
    if effects:
        write_csv(directory / "effects.csv", effects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", required=True, choices=("byte", "bot"))
    args = parser.parse_args()
    run(args.tracker)


if __name__ == "__main__":
    main()
