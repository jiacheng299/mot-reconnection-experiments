"""Extract deterministic, auditable examples of recovery gains and losses."""

import argparse
import json
from pathlib import Path

from reconnect_study.data_io import digest
from reconnect_study.study import save


def extract(trace_root, reference, candidate, output, per_direction=3):
    result = {"improved": [], "lost": []}
    reference_paths = sorted((trace_root / reference).glob("*.json"))
    candidate_paths = sorted((trace_root / candidate).glob("*.json"))
    if not reference_paths or {row.name for row in reference_paths} != {row.name for row in candidate_paths}:
        raise ValueError("Trace folders must contain the same nonempty video cohort")
    for path in reference_paths:
        other = trace_root / candidate / path.name
        first, second = (json.loads(source.read_bytes()) for source in (path, other))
        for a, b in zip(first["gap_records"], second["gap_records"], strict=True):
            indices = ("person", "before", "after")
            if any(a[key] != b[key] for key in indices):
                raise ValueError(f"Different gap definitions in {path.name}")
            if (a["outcome"] == "correct") == (b["outcome"] == "correct"):
                continue
            direction = "improved" if b["outcome"] == "correct" else "lost"
            if len(result[direction]) >= per_direction:
                continue
            result[direction].append({"sequence": path.stem, **{key: a[key] for key in indices},
                                      "reference_outcome": a["outcome"], "candidate_outcome": b["outcome"],
                                      "reference_trace_sha256": digest(path),
                                      "candidate_trace_sha256": digest(other)})
    save(output, {"reference": reference, "candidate": candidate, "cases": result,
                  "rule": "First recoveries gained and lost, sorted by source then evaluation event order."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.trace_root, args.reference, args.candidate, args.output)
