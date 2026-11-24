"""Audit completed acceptance calibration and export its locked work points."""

import argparse
import csv
from pathlib import Path
import subprocess

from reconnect_study.data_io import digest
from reconnect_study.policy import policy_specs
from reconnect_study.study import save
from reconnect_study.supplement_matched import ARCHIVE, BASELINES, CAPS, OUTPUT, load


def repo_revision(repository):
    command = ["git", "-C", str(repository), "rev-parse", "HEAD"]
    revision = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    changes = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {"commit": revision, "local_changes": changes}


def verify_frozen_baseline(key, directory):
    original = load(directory / "calibration-original.json")["by_video"]
    released = load(ARCHIVE / f"calibration-{key}-rows.json")["policies"]
    compared = 0
    for name, rows in original.items():
        if name not in released:
            continue
        archived = {row["name"]: row for row in released[name]}
        if set(archived) != {row["name"] for row in rows}:
            raise ValueError(f"B12 archived {name} uses a different cohort")
        for row in rows:
            for field in ("accepted", "wrong", "correct", "gaps", "unverified_accepted",
                          "ignored_accepted", "track_rows", "recovery_outcomes"):
                if field in archived[row["name"]] and row[field] != archived[row["name"]][field]:
                    raise ValueError(f"B12 archived {name} differs on {row['name']} {field}")
            compared += 1
    return compared


def export(key, root):
    inputs = load(root / "input-splits.json")
    expected_sources = {row["source_id"] for row in inputs["sequences"]
                        if row["split"] == "calibration"}
    work_points = []
    all_candidates = []
    checks = {}
    for cap in CAPS:
        directory = root / f"B{cap}"
        lock = load(directory / "locked.json")
        calibration_file = directory / lock["calibration_file"]
        if any(lock[field] != digest(path) for field, path in (
            ("config_sha256", directory / "config.json"),
            ("calibration_sha256", calibration_file),
            ("inputs_sha256", root / "input-splits.json"),
        )):
            raise ValueError(f"Lock checksum mismatch for B{cap}")
        retained = set(BASELINES)
        for label, point in lock["selection"]["points"].items():
            for family, row in point["nearest"].items():
                if point["matched"]:
                    retained.add(row["policy"])
                work_points.append({
                    "cap": cap, "target_name": label, "target": point["target"], "family": family,
                    "matched": point["matched"], "tolerance": point["tolerance"],
                    "maximum_family_spread": point["spread"], **row,
                })
        if retained != set(lock["test_policies"]):
            raise ValueError(f"Test list disagrees with work points for B{cap}")
        for grid in ("original", "dense"):
            path = directory / f"calibration-{grid}.json"
            if not path.exists():
                continue
            payload = load(path)
            grid_config_path = directory / f"config-{grid}.json"
            config = load(grid_config_path)
            if payload["config_sha256"] != digest(grid_config_path):
                raise ValueError(f"Calibration config hash mismatch for B{cap} {grid}")
            specs = policy_specs(config)
            for name, rows in payload["by_video"].items():
                if {row["source_id"] for row in rows} != expected_sources:
                    raise ValueError(f"Wrong calibration source roster: B{cap} {grid} {name}")
                threshold = specs[name][1] if name in specs else None
                aggregate = payload["aggregate"][name]
                counts = {field: aggregate.get(field, 0) for field in (
                    "gaps", "correct", "frames", "accepted", "ignored_accepted", "wrong",
                    "unverified_accepted",
                )}
                all_candidates.append({"cap": cap, "grid": grid, "policy": name, "threshold": threshold,
                                       **counts, **aggregate})
            checks[f"B{cap}_{grid}"] = payload["audit"]
        checks[f"B{cap}_policy_count"] = len(retained)
    checks["B12_paired_archived_rows_verified"] = verify_frozen_baseline(key, root / "B12")
    checks["vendor"] = {
        "byte": repo_revision(Path("vendor/ByteTrack")),
        "bot": repo_revision(Path("vendor/BoT-SORT")),
    }
    checks["exploratory_external_exclusion"] = {
        "videos": sum(row["split"] == "exploratory_external" for row in
                      load(ARCHIVE / "manifest.json")["sequences"]),
        "reason": "The manifest does not verify separate original recordings or sites for external mappings.",
    }
    for filename, rows in (("calibration-work-points.csv", work_points),
                           ("calibration-candidates.csv", all_candidates)):
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (root / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fields)
                writer.writeheader()
                writer.writerows(rows)
    save(root / "calibration-audit.json", checks)


def audit_test(key, root):
    inputs = load(root / "input-splits.json")
    expected = {row["name"]: (row["source_id"], row["frames"]) for row in inputs["sequences"]
                if row["split"] == "test"}
    frozen = load(ARCHIVE / f"test-{key}.json")
    checks = {}
    for cap in CAPS:
        directory = root / f"B{cap}"
        lock = load(directory / "locked.json")
        test_path = directory / "test.json"
        test = load(test_path)
        report = load(directory / "report.json")
        if (report["test_sha256"] != digest(test_path)
                or report["lock_sha256"] != digest(directory / "locked.json")
                or test["config_sha256"] != lock["config_sha256"]
                or test["benchmark_manifest_sha256"] != inputs["manifest_sha256"]):
            raise ValueError(f"Test provenance mismatch for B{cap}")
        if set(test["policies"]) != set(lock["test_policies"]):
            raise ValueError(f"Test policy roster mismatch for B{cap}")
        verified = 0
        for name, rows in test["policies"].items():
            actual = {row["name"]: (row["source_id"], row["frames"]) for row in rows}
            if actual != expected or len(rows) != len(expected):
                raise ValueError(f"Test cohort changed for B{cap} {name}")
            for row in rows:
                trace = directory / "traces" / name / f"{row['name']}.json"
                if digest(trace) != row["trace_sha256"]:
                    raise ValueError(f"Trace checksum mismatch: {trace}")
                if not 0 <= row["unverified_accepted"] <= row["wrong"] <= row["accepted"]:
                    raise ValueError(f"Inconsistent event counts for {trace}")
                verified += 1
            if name == "unconstrained_tracker" or (cap == 12 and name == "cap_only"):
                historical = {row["name"]: row for row in frozen["policies"][name]}
                for row in rows:
                    old = historical[row["name"]]
                    for field in ("accepted", "wrong", "correct", "gaps", "unverified_accepted",
                                  "ignored_accepted", "track_rows", "recovery_outcomes", "trace_sha256"):
                        if field in old and row[field] != old[field]:
                            raise ValueError(f"Archived test baseline changed: B{cap} {name} {row['name']} {field}")
                if test["overall"][name]["aggregate"] != frozen["overall"][name]["aggregate"]:
                    raise ValueError(f"Archived overall baseline changed: B{cap} {name}")
        checks[f"B{cap}"] = {"videos_per_policy": len(expected), "policy_count": len(test["policies"]),
                            "verified_trace_digests": verified, "test_sha256": digest(test_path),
                            "report_sha256": digest(directory / "report.json")}
    save(root / "test-integrity.json", checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", required=True, choices=("byte", "bot"))
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        audit_test(args.tracker, OUTPUT / args.tracker)
    else:
        export(args.tracker, OUTPUT / args.tracker)


if __name__ == "__main__":
    main()
