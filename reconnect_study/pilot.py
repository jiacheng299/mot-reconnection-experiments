"""Fit reconnect scores and compare them by closed-loop exploratory replay."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from reconnect_study.data_io import read_manifest
from reconnect_study.policy import assignment, margin
from reconnect_study.replay import load_sequence, replay
from reconnect_study.scoring import fit


QUANTILES = (.25, .50, .70, .85, .93)


def pilot_split(records):
    sources = {row["source_id"] for row in records}
    ordered = sorted(sources, key=lambda x: hashlib.sha256(("reconnect-2026-pilot:" + x).encode())
                     .hexdigest())
    return set(ordered[:len(ordered) // 2])


def auc(model, rows):
    labels = np.array([row[2] for row in rows], dtype=bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = rankdata([model(row[1]) for row in rows])
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def counts(rows):
    result = {key: sum(row.get(key, 0) for row in rows) for key in
              ("frames", "gaps", "correct", "accepted", "wrong", "unverified_accepted",
               "ignored_accepted")}
    result["wrong_per_acceptance"] = result["wrong"] / result["accepted"] if result["accepted"] else None
    result["correct_per_gap"] = result["correct"] / result["gaps"] if result["gaps"] else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantiles", type=float, nargs="+", default=QUANTILES)
    args = parser.parse_args()
    records = json.loads(args.baseline.read_bytes())["sequences"]
    develop = pilot_split(records)
    rows = defaultdict(list)
    for record in records:
        source = record["source_id"]
        key = "development" if source in develop else "evaluation"
        rows[key].extend((source, action["features"], action["correct"]) for action in record["actions"])
    scores = {"assignment": assignment, "margin": margin}
    training = {}
    for name, structured, rank in (("linear", False, 0.0), ("structured", True, 0.0),
                                   ("structured_rank", True, 0.35)):
        model, summary = fit(rows["development"], structured=structured,
                             ranking_weight=rank, device=args.device)
        scores[name] = model
        training[name] = {"model": asdict(model), **summary}

    scores_on_baseline_proposals = {name: auc(model, rows["evaluation"]) for name, model in scores.items()}
    thresholds = {name: np.unique(np.quantile([model(row[1]) for row in rows["development"]],
                                              args.quantiles)).tolist()
                  for name, model in scores.items()}
    evaluation = [row for row in records if row["source_id"] not in develop]
    by_policy = defaultdict(list)
    args.output.mkdir(parents=True, exist_ok=True)
    setup = {"development_sources": sorted(develop), "evaluation_sources": sorted(
        {row["source_id"] for row in evaluation}), "training": training, "auc": scores_on_baseline_proposals,
        "thresholds": thresholds, "quantiles": args.quantiles, "reference": counts(evaluation)}
    (args.output / "setup.json").write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
    print("Frozen exploratory configuration", setup["auc"], flush=True)

    for sequence in read_manifest(args.manifest):
        if sequence.source_id in develop:
            continue
        detections, evidence = load_sequence(sequence)
        results = {}
        for name, model in scores.items():
            for index, threshold in enumerate(thresholds[name]):
                def gate(proposal):
                    return model(proposal.features) >= threshold

                record, _ = replay(sequence, detections, evidence, gate)
                key = f"{name}_{index}"
                by_policy[key].append(record)
                results[key] = record
        (args.output / f"{sequence.name}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                            encoding="utf-8")
        print(sequence.name, len(results), "complete", flush=True)
    aggregate = {key: counts(records) for key, records in by_policy.items()}
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
