"""Run shared-input tracking and TrackEval over the official DanceTrack val split."""

import argparse
from functools import partial
import json
from math import fsum
from pathlib import Path

import numpy as np

from scripts.detect_mot17_dev import file_sha256
from scripts.p5_dance_data import DANCE, sequences, verify_cache
from scripts.run_p1 import load_tracker, predict, read_detections
from scripts.score_p3_c1 import OPTIONS, track_sequence


GATES = {'wide': (0.25, 0.45), 'reference': (0.30, 0.50), 'tight': (0.35, 0.55)}


def set_gate(profile):
    OPTIONS['strong_box_iou'], OPTIONS['weak_box_iou'] = GATES[profile]


def hand_score(group, action, legal, _confidence, weight=0.10):
    overlaps = [legal[track].get(det, 0.0) for track, det in zip(group, action, strict=True)]
    pair_penalty = fsum(abs(overlaps[i] - overlaps[j])
                        for i, track in enumerate(group) for j in range(i + 1, len(group))
                        if action[i] is not None and action[j] is not None
                        and legal[track].keys() & legal[group[j]].keys())
    return fsum(overlaps) - weight * pair_penalty


def cached_detections(path, split, sequence, length, manifest_check=verify_cache):
    record = manifest_check(path, split, sequence, length)
    detections = read_detections(path, length)
    if sum(map(len, detections.values())) != record['count']:
        raise ValueError(f'Cache record count differs from {path}')
    return record, detections


def score_trackers(run, split, lengths, methods):
    import trackeval

    evaluator = trackeval.Evaluator({'USE_PARALLEL': False, 'PLOT_CURVES': False})
    dataset = trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER': str(DANCE / split), 'TRACKERS_FOLDER': str(run / 'trackers'),
        'OUTPUT_FOLDER': str(run / 'trackeval'), 'TRACKERS_TO_EVAL': methods,
        'BENCHMARK': 'DanceTrack', 'SPLIT_TO_EVAL': split, 'SKIP_SPLIT_FOL': True,
        'SEQ_INFO': lengths, 'DO_PREPROC': True,
    })
    evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])


def run_split(run, cache, split, method, ranker=None, candidate_limit=16, only=None, timings=False, checkpoint=None):
    selected = [(folder, info) for folder, info in sequences(split) if only is None or folder.name == only]
    if not selected:
        raise ValueError(f'No matching {split} sequence ({only})')
    run.mkdir(parents=True, exist_ok=False)
    lengths, manifests, metrics = {}, {}, {}
    byte = load_tracker(run) if method == 'ByteTrack' else None
    for folder, info in selected:
        name, length = folder.name, info.getint('seqLength')
        path = cache / split / f'{name}.txt'
        record, detections = cached_detections(path, split, name, length)
        manifests[name] = record['detections_sha256']
        lengths[name] = length
        if byte:
            from yolox.tracker.basetrack import BaseTrack

            BaseTrack._count = 0
            predict(run, info, source=path, tracker_class=byte)
        else:
            frames = [] if timings else None
            summary = track_sequence(
                run, detections, length, sequence=name, method=method, ranker=ranker,
                candidate_limit=candidate_limit, audit_path=run / 'associations' / f'{name}.jsonl', timings=frames
            )
            summary['detections_sha256'] = record['detections_sha256']
            (run / 'associations' / f'{name}.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
            if frames is not None:
                metrics[name] = frames
        print(f'{method} {split}: {len(lengths)}/{len(selected)} {name} ({length} frames)', flush=True)
    if metrics:
        (run / 'association-timings.json').write_text(json.dumps(metrics), encoding='utf-8')
        values = np.concatenate([np.asarray(frames) for frames in metrics.values()])
        print(json.dumps({'average_ms': values[:, 0].mean() / 1e6,
                          'p95_ms': np.percentile(values[:, 0], 95) / 1e6,
                          'p99_ms': np.percentile(values[:, 0], 99) / 1e6,
                          'candidates_per_group': values[:, 1].sum() / max(1, values[:, 2].sum())}), flush=True)
    (run / 'input-manifest.json').write_text(json.dumps({
        'split': split, 'method': method, 'candidate_limit': candidate_limit, 'gate_options': OPTIONS,
        'coverage': lengths,
        'detections_sha256': manifests,
        'checkpoint_sha256': file_sha256(checkpoint) if checkpoint else None,
    }, indent=2) + '\n', encoding='utf-8')
    if split == 'val':
        score_trackers(run, split, lengths, [method])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--split', default='val', choices=('train', 'val'))
    parser.add_argument('--method', default='C1', choices=('ByteTrack', 'C1', 'hand'))
    parser.add_argument('--hand-weight', type=float, choices=(0.05, 0.10, 0.20), default=0.10)
    parser.add_argument('--gate', choices=tuple(GATES), default='reference')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--sequence')
    parser.add_argument('--k', default=16, type=int, choices=(1, 4, 8, 16, 32))
    parser.add_argument('--timings', action='store_true')
    args = parser.parse_args()
    set_gate(args.gate)
    ranker = partial(hand_score, weight=args.hand_weight) if args.method == 'hand' else None
    method = args.method
    if args.checkpoint:
        from scripts.p5_train import load_ranker

        ranker, method = load_ranker(args.checkpoint)
    run_split(args.run, args.cache, args.split, method, ranker, args.k, args.sequence, args.timings, args.checkpoint)
