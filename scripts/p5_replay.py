"""Build sharded action-consistent replay from the entire DanceTrack training split."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.detect_mot17_dev import DATA, file_sha256, sequence_info
from scripts.p4_joint import encode_group, tracking_context
from scripts.p4_supervision import current_quality, load_pedestrian_truth, matched_people
from scripts.p4_td import independent_successor, replay_actions
from scripts.p5_dance_data import sequences
from scripts.p5_dance_eval import GATES, cached_detections, set_gate
from scripts.p5_mot_detector import TRAIN as MOT_TRAIN, verify_cache as verify_mot_cache
from scripts.run_p1 import load_tracker
from scripts.score_p3_c1 import OPTIONS, commit_frame, prepare_frame, select_frame_actions


SAMPLE_STRIDE = 4
COLLECTION_SEED = 0


def emit_transitions(active, groups, boxes, legal, anchors, identities, frame, kalman,
                     next_detections, branch_rng, preference_rng):
    confidence = {det: box.score for det, box in boxes.items()}
    context = tracking_context(active, frame)
    ratings, encoded = [], []
    for group in groups:
        tracks, actions = group['tracks'], group['candidates']
        ratings.append([current_quality(tracks, a, legal, anchors, identities) / len(tracks) for a in actions])
        encoded.append([encode_group(tracks, a, legal, confidence, context) for a in actions])
    eligible = [i for i, labels in enumerate(ratings) if min(labels) != max(labels)]
    for change, action in replay_actions(groups, branch_rng):
        choices = [g['candidates'].index(action if g is change else g['selected']) for g in groups]
        if change is not None:
            focus = groups.index(change)
            preferred = focus if focus in eligible else None
        else:
            preferred = preference_rng.choice(eligible) if eligible else None
        pair = []
        if preferred is not None:
            labels = ratings[preferred]
            best = preference_rng.choice([i for i, value in enumerate(labels) if value == max(labels)])
            worse = preference_rng.choice([i for i, value in enumerate(labels) if value < max(labels)])
            pair = [encoded[preferred][best], encoded[preferred][worse]]
        selected = {track: det for g, i in zip(groups, choices, strict=True)
                    for track, det in zip(g['tracks'], g['candidates'][i], strict=True)}
        _, successor = independent_successor(active, boxes, selected, frame, kalman, next_detections, joint=True)
        yield {'frame': frame, 'terminal': next_detections is None,
               'joint': [scores[index] for scores, index in zip(encoded, choices, strict=True)],
               'preference': pair, 'reward': sum(scores[index] for scores, index in zip(ratings, choices, strict=True)),
               'successor': [{'joint': group['joint']} for group in successor]}


def collect_sequence(output, folder, info, cache, dataset='DanceTrack'):
    from yolox.tracker.basetrack import BaseTrack
    from yolox.tracker.kalman_filter import KalmanFilter

    name, length = folder.name, info.getint('seqLength')
    cache_file = cache / 'train' / f'{name}.txt'
    if dataset == 'MOT17':
        manifest, detections = cached_detections(cache_file, 'train', name, length, verify_mot_cache)
    else:
        manifest, detections = cached_detections(cache_file, 'train', name, length)
    path = output / f'{name}.jsonl.gz'
    meta = output / f'{name}.json'
    gt = folder / 'gt' / 'gt.txt'
    label_hash = file_sha256(gt)
    if path.is_file() and meta.is_file():
        recorded = json.loads(meta.read_bytes())
        if (recorded['cache_sha256'] == manifest['detections_sha256'] and recorded['gt_sha256'] == label_hash and
                recorded['sha256'] == file_sha256(path) and recorded['stride'] == SAMPLE_STRIDE and
                recorded['gate_options'] == OPTIONS):
            return recorded
        raise ValueError(f'Existing replay shard differs from {name}')
    people = load_pedestrian_truth(gt)
    BaseTrack._count = 0
    sequence_index = int(name.split('-')[1] if dataset == 'MOT17' else name.removeprefix('dancetrack'))
    branch_rng, preference_rng = [np.random.default_rng(seed)
                                  for seed in np.random.SeedSequence([COLLECTION_SEED, sequence_index]).spawn(2)]
    kalman, active, anchors = KalmanFilter(), {}, {}
    counts = Counter()
    staged = path.with_suffix('.tmp')
    with staged.open('wb') as file, gzip.GzipFile(filename='', mode='wb', fileobj=file, mtime=0, compresslevel=3) as stream:
        for frame in range(1, length + 1):
            boxes, legal = prepare_frame(active, detections[frame])
            assigned, groups, _ = select_frame_actions(legal)
            identities = matched_people(detections[frame], people[frame])
            if frame == length or (frame - 1) % SAMPLE_STRIDE == 0:
                counts['sampled_frames'] += 1
                following = detections[frame + 1] if frame < length else None
                for record in emit_transitions(active, groups, boxes, legal, anchors, identities, frame,
                                               kalman, following, branch_rng, preference_rng):
                    stream.write((json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode())
                    counts['rows'] += 1
                    counts['terminal_rows'] += record['terminal']
                    counts['pref_rows'] += bool(record['preference'])
            births, *_ = commit_frame(active, boxes, assigned, frame, kalman)
            for track, det in [*assigned.items(), *births]:
                if det in identities:
                    anchors.setdefault(track, identities[det])
    staged.replace(path)
    result = {'dataset': dataset, 'sequence': name, 'split': 'train', 'frames': length,
              'stride': SAMPLE_STRIDE, 'collection_seed': COLLECTION_SEED, 'cache_sha256': manifest['detections_sha256'],
              'gt_sha256': label_hash, 'sha256': file_sha256(path), 'gate_options': OPTIONS, **counts}
    meta.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def collect_all(output, cache, sequence=None, dataset='DanceTrack'):
    output.mkdir(parents=True, exist_ok=True)
    load_tracker(output / 'workers' / (sequence or 'all'))
    inputs = ((DATA / 'gt' / name, sequence_info(name)) for name in MOT_TRAIN) if dataset == 'MOT17' else sequences('train')
    found = False
    for folder, info in inputs:
        if sequence is None or folder.name == sequence:
            found = True
            print(json.dumps(collect_sequence(output, folder, info, cache, dataset)), flush=True)
    if not found:
        raise ValueError(f'{sequence} is not in {dataset} training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--sequence')
    parser.add_argument('--gate', choices=tuple(GATES), default='reference')
    parser.add_argument('--dataset', choices=('DanceTrack', 'MOT17'), default='DanceTrack')
    args = parser.parse_args()
    set_gate(args.gate)
    collect_all(args.output, args.cache, args.sequence, args.dataset)
