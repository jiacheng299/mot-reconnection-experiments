"""Replay a frozen BoostTrack-v1 adapter lock on the archived test cohort."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from reconnect_study import study
from reconnect_study.data_io import digest, read_manifest
from reconnect_study.round3_ablation import compare_match
from reconnect_study.round3_ablation_eval import bound, verify_sources
from reconnect_study.round3_boost_bot import run_botsort_boost
from reconnect_study.round3_boost_calibrate import ARCHIVE, CAP, OUTPUT, ROUND2, score, tracker
from reconnect_study.supplement_matched import load


def run_policy(task):
    config, policy, path, output = task
    study.TRACKERS[config["tracker"]] = tracker if config["tracker"] == "bytetrack" else run_botsort_boost
    study.policy_specs = lambda _: {policy: (score, config["thresholds"][policy])}
    study.run_split(ARCHIVE / "manifest.json", output / "config.json", "test", path,
                    [policy], "personpath22-native-keyframe", None, output / "traces")
    return load(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("byte", "bot"), default="byte")
    key = parser.parse_args().host
    output = OUTPUT if key == "byte" else OUTPUT / "bot"
    old = ROUND2 if key == "byte" else ROUND2.parent.parent / "bot" / "B12"
    lock_path = output / "locked.json"
    lock = load(lock_path)
    config_path = output / "config.json"
    calibration_path = output / "calibration.json"
    bound(config_path, lock["config_sha256"])
    bound(calibration_path, lock["calibration_sha256"])
    bound(old / "locked.json", lock["round2_lock_sha256"])
    config, calibration = load(config_path), load(calibration_path)
    manifest_path = ARCHIVE / "manifest.json"
    bound(manifest_path, config["study_manifest_sha256"])
    bound(output / "development-numeric-audit.json", config["numeric_audit_sha256"])
    for source, sha256 in config.get("source_sha256", {}).items():
        bound(source, sha256)
    bound(config_path, calibration["config_sha256"])
    if (config["round2_lock_sha256"] != lock["round2_lock_sha256"] or config["acceptance_cap"] != CAP
            or lock["acceptance_cap"] != CAP or calibration["selection"] != lock["selection"]):
        raise ValueError("Inconsistent BoostTrack lock, config or calibration")
    prior = load(OUTPUT.parent / "ablation" / key / "locked.json")["work_points"]
    destination = output / "test.json"
    if destination.exists():
        raise FileExistsError(destination)
    comparisons = {}
    for point, external in lock["selection"].items():
        reference = prior[point]
        if (external["target"] != reference["target"] or external["tolerance"] != reference["tolerance"]
                or external["threshold"] != config["thresholds"][external["policy"]]):
            raise ValueError(f"Inconsistent target or threshold for {point}")
        match = compare_match(reference["old"]["structured"]["mean_accepted"], external["mean_accepted"],
                              external["target"], external["tolerance"])
        comparisons[point] = {**external, "file": str(destination), "calibration_matched": match["matched"],
                              "calibration_comparison": match}
    policies = lock["test_policies"]
    if sorted({entry["policy"] for entry in comparisons.values() if entry["calibration_matched"]}) != policies:
        raise ValueError("External policies do not match locked pairwise comparisons")
    summary_path = OUTPUT / f"{key}-summary-input.json"
    if not policies:
        study.save(summary_path, comparisons)
        return
    expected = {(row.name, row.source_id, row.frames) for row in read_manifest(manifest_path) if row.split == "test"}
    if len(expected) != 74 or len({row[1] for row in expected}) != 74:
        raise ValueError("Unexpected test cohort")
    with TemporaryDirectory(dir=output) as scratch, ProcessPoolExecutor(max_workers=len(policies)) as executor:
        tasks = [(config, policy, Path(scratch) / f"{policy}.json", output) for policy in policies]
        runs = list(executor.map(run_policy, tasks))
    result = {**runs[0], "lock_sha256": digest(lock_path)}
    for field in ("policies", "aggregate", "overall"):
        result[field] = {key: value for run in runs for key, value in run[field].items()}
    verify_sources(result, expected)
    for policy, rows in result["policies"].items():
        for row in rows:
            bound(output / "traces" / policy / f"{row['name']}.json", row["trace_sha256"])
    study.save(destination, result)
    study.save(summary_path, comparisons)


if __name__ == "__main__":
    main()
