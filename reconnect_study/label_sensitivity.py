"""Re-score saved test decisions under a duplicate-box equivalence rule."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.events import evaluate_prepared
from reconnect_study.replay import load_sequence
from reconnect_study.study import save


def recount(manifest, test, traces, output, selected):
    evidence_by_policy = {name: [] for name in selected}
    result = json.loads(test.read_bytes())
    source_rows = {name: {row["name"]: row for row in result["policies"][name]} for name in selected}
    for sequence in read_manifest(manifest):
        if sequence.split != "test":
            continue
        _, evidence = load_sequence(sequence)
        for method in selected:
            path = traces / method / f"{sequence.name}.json"
            expected = source_rows[method][sequence.name]
            if expected["trace_sha256"] != digest(path):
                raise ValueError(f"Trace differs from the tested output: {path}")
            trace = json.loads(path.read_bytes())
            tracks = [SimpleNamespace(**row) for row in trace["tracks"]]
            actions = [SimpleNamespace(proposal=SimpleNamespace(**row["proposal"]), accepted=row["accepted"])
                       for row in trace["decisions"] if row["accepted"]]
            alternate = evaluate_prepared(evidence, tracks, actions, duplicate_equivalence=True)
            if alternate.gaps != expected["gaps"] or alternate.correct != expected["correct"]:
                raise ValueError("The sensitivity analysis changed the recovery outcome")
            evidence_by_policy[method].append({
                "name": sequence.name, "strict_accepted": expected["accepted"], "strict_wrong": expected["wrong"],
                "alternate_accepted": alternate.accepted, "alternate_wrong": alternate.wrong})
    keys = ("strict_accepted", "strict_wrong", "alternate_accepted", "alternate_wrong")
    save(output, {"scope": "Post hoc sensitivity on fixed recorded outputs; thresholds and actions unchanged.",
                  "rule": "Unassigned boxes within 0.001 px of a box assigned to a single ground-truth ID "
                          "inherit that evaluation identity.",
                  "sequences": evidence_by_policy,
                  "aggregate": {name: {key: sum(row[key] for row in values) for key in keys}
                                for name, values in evidence_by_policy.items()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("policies", nargs="+")
    args = parser.parse_args()
    recount(args.manifest, args.test, args.traces, args.output, args.policies)
