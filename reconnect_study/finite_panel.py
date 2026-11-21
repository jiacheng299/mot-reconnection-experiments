"""Combine frozen policy replays for a prespecified random panel of sources."""

import argparse
import json
from pathlib import Path

from reconnect_study.data_io import digest
from reconnect_study.study import save


def combine(design_path, config_path, manifest_path, inputs, output):
    design = json.loads(design_path.read_bytes())
    if (design["config_sha256"] != digest(config_path)
            or design["benchmark_manifest_sha256"] != digest(manifest_path)):
        raise ValueError("The draw refers to a different set of frozen inputs")
    config = json.loads(config_path.read_bytes())
    names = {f"{method}_{index}" for method, thresholds in config["thresholds"].items()
             for index in range(len(thresholds))}
    selected = set(design["calibration_sample"])
    policies = {name: [] for name in sorted(names)}
    for path in inputs:
        replay = json.loads(path.read_bytes())
        if (replay["config_sha256"] != design["config_sha256"]
                or replay["acceptance_cap"] != design["acceptance_cap"]
                or replay["tracker"] != design["tracker"]):
            raise ValueError(f"Changed replay inputs: {path}")
        if set(replay["policies"]) - {"unconstrained_tracker"} != names:
            raise ValueError(f"Changed frozen policy family: {path}")
        for name in names:
            policies[name].extend(row for row in replay["policies"][name]
                                  if row["source_id"] in selected)
    for name, rows in policies.items():
        if {row["source_id"] for row in rows} != selected or len(rows) != len({row["name"] for row in rows}):
            raise ValueError(f"Missing source or duplicate video in {name}")
    out_path = Path(output)
    replay_path = out_path.with_name(out_path.stem + "-rows.json")
    save(replay_path, {"tracker": design["tracker"], "config_sha256": design["config_sha256"],
                       "split": "finite_calibration",
                       "acceptance_cap": design["acceptance_cap"], "policies": policies,
                       "input_digests": {str(path): digest(path) for path in inputs}})
    save(out_path, {
        "finite_population_design": str(design_path.resolve()),
        "frozen_config": str(config_path.resolve()),
        "benchmark_manifest": str(manifest_path.resolve()),
        "policy_rows": replay_path.name,
        "include_policies": sorted(names),
        "cohort": {"description": "Prespecified uniform calibration subset of a fixed PersonPath22 panel.",
                   "iid_source_sampling": False, "policies_frozen_before_calibration": True,
                   "caveats": ["The certificate applies to the full fixed benchmark panel."]},
        "event_weighted_cap": {"max_accepted": design["acceptance_cap"],
                               "enforcement": "Replay gate counts accepted actions by source and stops at cap.",
                               "enforced_in_replay_and_deployment": True},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("replays", nargs="+", type=Path)
    args = parser.parse_args()
    combine(args.design, args.config, args.manifest, args.replays, args.output)
