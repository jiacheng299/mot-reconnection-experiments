"""Evaluate locked finite-panel policies on the random held-out complement."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from reconnect_study.data_io import digest
from reconnect_study.pilot import counts
from reconnect_study.study import save


def evaluate(design_path, certificate_path, input_paths, output):
    design = json.loads(design_path.read_bytes())
    certificate = json.loads(certificate_path.read_bytes())
    target_sources = set(design["held_out_complement"])
    selected = {entry["selected_policy"] for entry in certificate["methods"].values()
                if entry["selected_policy"]}
    policies = {}
    for path in input_paths:
        payload = json.loads(path.read_bytes())
        if payload["config_sha256"] != design["config_sha256"]:
            raise ValueError("Replay used a different frozen configuration")
        for name, records in payload["policies"].items():
            policies.setdefault(name, []).extend(row for row in records if row["source_id"] in target_sources)
    evaluation = {}
    for name, rows in policies.items():
        actual_sources = {row["source_id"] for row in rows}
        if actual_sources != target_sources:
            continue
        if len(rows) != len({row["name"] for row in rows}):
            raise ValueError(f"Duplicated video for {name}")
        source_counts = defaultdict(lambda: [0, 0])
        for row in rows:
            aggregate = source_counts[row["source_id"]]
            aggregate[0] += row["wrong"]
            aggregate[1] += row["accepted"]
        active = [wrong / accepted for wrong, accepted in source_counts.values() if accepted]
        evidence = {
            **counts(rows), "sources": len(actual_sources),
            "active_sources": len(active),
            "active_video_observed_risk": sum(active) / len(active) if active else None,
            "certificate_selected": name in selected,
        }
        evaluation[name] = evidence
    if not selected.issubset(evaluation):
        raise ValueError("Missing held-out data for a certified selected policy")
    save(output, {"scope": "Held-out finite-panel complement; descriptive utility only.",
                  "source_count": len(target_sources),
                  "design_sha256": digest(design_path), "certificate_sha256": digest(certificate_path),
                  "selected_policies": sorted(selected), "aggregate": evaluation})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("replays", type=Path, nargs="+")
    args = parser.parse_args()
    evaluate(args.design, args.certificate, args.replays, args.output)
