"""E5 fixed-state and E6 competition/visibility diagnostics under ByteTrack."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from functools import partial
import gzip
import json
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np

from scripts.detect_mot17_dev import DATA, SEQUENCE, file_sha256, sequence_info
from scripts.p3_association import competition_groups
from scripts.p3_c1 import choose_group_action
from scripts.p4_joint import tracking_context
from scripts.p4_supervision import current_quality, load_pedestrian_truth, matched_people
from scripts.p5_dance_data import sequences
from scripts.p5_dance_eval import GATES, cached_detections, hand_score, set_gate
from scripts.p5_train import load_ranker
from scripts.run_p1 import load_tracker, read_detections
from scripts.score_p3_c1 import prepare_frame


def division(size):
    return '1' if size == 1 else '2–4' if size <= 4 else '5–8' if size <= 8 else '≥9'


def visibility_rows(gt):
    result = defaultdict(dict)
    with gt.open(newline='') as file:
        for row in csv.reader(file):
            if row[6:8] == ['1', '1']:
                result[int(row[0])][int(row[1])] = float(row[8])
    return result


def visibility_bin(value):
    if value < 0 or value > 1:
        return 'unavailable'
    return '<0.3' if value < 0.3 else '[0.3,0.7)' if value < 0.7 else '≥0.7'


def evaluate_decisions(tracks, action, legal, identities, anchors, frame_truth):
    counts = Counter()
    for track, detection in zip(tracks, action, strict=True):
        person = anchors.get(track)
        if person is None or person not in frame_truth:
            continue
        counts['visible_identity'] += 1
        if not any(identities.get(det) == person for det in legal[track]):
            counts['gate_unavailable'] += 1
            continue
        counts['eligible_continuations'] += 1
        if identities.get(detection) == person:
            counts['correct'] += 1
        elif detection is None:
            counts['miss'] += 1
        else:
            counts['wrong'] += 1
    return counts


def update_stats(stats, used, choice, oracle, elapsed, size, contested, candidates, score):
    stats['groups'] += 1
    stats['tracks'] += size
    stats['contested_groups'] += contested > 0
    stats['contested_detections'] += contested
    stats['legal_detections'] += used
    stats['candidates'] += candidates
    stats['oracle_optimal_choices'] += choice == oracle
    stats['reward_sum'] += score
    stats['latency_ns'].append(elapsed)


def audit_sequence(tracker_class, detections, info, gt, joint, hand_weight, budgets, stats, mot=False, trace=None):
    from yolox.tracker.basetrack import BaseTrack

    length = info.getint('seqLength')
    size = (info.getint('imHeight'), info.getint('imWidth'))
    truth = load_pedestrian_truth(gt)
    visible = visibility_rows(gt) if mot else {}
    BaseTrack._count = 0
    tracker = tracker_class(SimpleNamespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False),
                            frame_rate=info.getint('frameRate'))
    anchors = {}
    hand = partial(hand_score, weight=hand_weight)
    for frame in range(1, length + 1):
        raw = detections[frame]
        active = {track.track_id: track for track in [*tracker.tracked_stracks, *tracker.lost_stracks]}
        snapshot = deepcopy(active)
        boxes, legal = prepare_frame(snapshot, raw)
        context = tracking_context(snapshot, frame)
        confidence = {index: box.score for index, box in boxes.items()}
        identities = matched_people(raw, truth[frame])
        audit = []
        for group in competition_groups(legal):
            popularity = Counter(det for track in group for det in legal[track])
            contested = sum(count > 1 for count in popularity.values())
            group_bin = division(len(group))
            for budget in budgets:
                start = perf_counter_ns()
                baseline, candidates, failed = choose_group_action(group, legal, budget)
                generation = perf_counter_ns() - start
                start = perf_counter_ns()
                chosen_hand = baseline if failed else max(candidates, key=lambda a: hand(group, a, legal, confidence))
                elapsed_hand = perf_counter_ns() - start
                start = perf_counter_ns()
                learned = baseline if failed or len(candidates) == 1 else candidates[
                    int(np.argmax(joint.score_candidates(group, candidates, legal, confidence, context)))
                ]
                elapsed_joint = perf_counter_ns() - start
                oracle = max(current_quality(group, act, legal, anchors, identities) for act in candidates)
                if trace is not None:
                    audit.append({'group': group, 'budget': budget, 'candidates': candidates,
                                  'selected': [baseline, chosen_hand, learned]})
                for label, act, elapsed in (('geometry', baseline, 0), ('manual-pair', chosen_hand, elapsed_hand),
                                            ('learned', learned, elapsed_joint)):
                    key = (budget, label, group_bin)
                    bucket = stats[key]
                    policy_reward = current_quality(group, act, legal, anchors, identities)
                    update_stats(bucket, len(popularity), policy_reward, oracle, generation + elapsed, len(group),
                                 contested, len(candidates), policy_reward / len(group))
                    bucket.update(evaluate_decisions(group, act, legal, identities, anchors, truth[frame]))
                    if mot and budget == 16:
                        for track, det in zip(group, act, strict=True):
                            person = anchors.get(track)
                            if person not in visible.get(frame, {}):
                                continue
                            label_vis = visibility_bin(visible[frame][person])
                            stats[(budget, label, f'visibility {label_vis}')].update(
                                evaluate_decisions((track,), (det,), legal, identities, anchors, truth[frame])
                            )
        if trace is not None:
            trace.write(json.dumps({'frame': frame, 'legal': legal, 'context': context, 'confidence': confidence,
                                    'decisions': audit}, separators=(',', ':'), allow_nan=False) + '\n')
        tracks = tracker.update(np.asarray(raw, dtype=float).reshape(-1, 5), size, size)
        outputs = [(*track.tlbr, track.score) for track in tracks]
        output_ids = matched_people(outputs, truth[frame])
        for index, person in output_ids.items():
            anchors.setdefault(tracks[index - 1].track_id, person)


def finalize(stats):
    output = []
    for (k, scorer, bin_name), row in stats.items():
        latency = np.asarray(row.pop('latency_ns'))
        count = row['eligible_continuations']
        item = {'k': k, 'scorer': scorer, 'bin': bin_name, **row,
                'continuation_accuracy': row['correct'] / count if count else None}
        if latency.size:
            item.update({'mean_ms': float(latency.mean() / 1e6), 'p95_ms': float(np.percentile(latency, 95) / 1e6),
                         'contested_detection_rate': row['contested_detections'] / max(1, row['legal_detections']),
                         'candidate_rate': row['candidates'] / row['groups'],
                         'oracle_choice_rate': row['oracle_optimal_choices'] / row['groups']})
        output.append(item)
    return output


def run(output, cache, checkpoint, gate, weight, mot_cache=None):
    set_gate(gate)
    joint, method = load_ranker(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    tracker = load_tracker(output)
    stats = defaultdict(lambda: Counter(latency_ns=[]))
    inputs = {}
    traces = {}
    budgets = (16,) if mot_cache else (1, 4, 8, 16, 32)
    for budget in budgets:
        for scorer in ('geometry', 'manual-pair', 'learned'):
            bins = ('1', '2–4', '5–8', '≥9')
            if mot_cache:
                bins += ('visibility <0.3', 'visibility [0.3,0.7)', 'visibility ≥0.7', 'visibility unavailable')
            for label in bins:
                stats[(budget, scorer, label)]['eligible_continuations'] = 0

    def collect(name, detections, info, gt, mot=False):
        path = output / f'{name}.jsonl.gz'
        with gzip.open(path, 'wt', encoding='utf-8', compresslevel=3) as trace:
            audit_sequence(tracker, detections, info, gt, joint, weight, budgets, stats, mot=mot, trace=trace)
        traces[name] = file_sha256(path)

    if mot_cache:
        info = sequence_info()
        gt = DATA / 'gt' / SEQUENCE / 'gt' / 'gt.txt'
        detections = read_detections(mot_cache, info.getint('seqLength'))
        collect(SEQUENCE, detections, info, gt, mot=True)
        inputs[SEQUENCE] = file_sha256(mot_cache)
    else:
        for folder, info in sequences('val'):
            cache_file = cache / 'val' / f'{folder.name}.txt'
            manifest, detections = cached_detections(cache_file, 'val', folder.name, info.getint('seqLength'))
            collect(folder.name, detections, info, folder / 'gt' / 'gt.txt')
            inputs[folder.name] = manifest['detections_sha256']
            print(f'Diagnosed {len(inputs)}/25 {folder.name}', flush=True)
    file = output / 'scene-diagnostics.tmp'
    file.write_text(json.dumps({'reference': 'ByteTrack', 'model': method, 'gate': gate,
                                'scoring_checkpoint_sha256': file_sha256(checkpoint), 'inputs': inputs,
                                'reference_trace_sha256': traces,
                                'rows': finalize(stats)}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    file.replace(file.with_suffix('.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--gate', choices=GATES, required=True)
    parser.add_argument('--hand-weight', type=float, required=True)
    parser.add_argument('--mot-cache', type=Path)
    args = parser.parse_args()
    run(args.output, args.cache, args.checkpoint, args.gate, args.hand_weight, args.mot_cache)
