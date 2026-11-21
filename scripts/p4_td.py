"""Collect action-consistent replay and fit finite-batch TD controls for P4."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from scripts.detect_mot17_dev import DATA, file_sha256, sequence_info
from scripts.p4_supervision import (
    FEATURES, REGULARIZATION, TRAIN_SEQUENCE, candidate_features, current_quality,
    fit_immediate_rewards, load_pedestrian_truth, matched_people,
)
from scripts.run_p1 import load_tracker, read_detections
from scripts.score_p3_c1 import commit_frame, prepare_frame, select_frame_actions


GAMMAS = (0.0, 0.9, 0.99)
UPDATE_COUNT = 20
REPLAY_SEED = 0
ALTERNATIVES_PER_FRAME = 3


def next_frame_features(groups, search_weights, dimensions):
    total = np.zeros(dimensions)
    for candidates in groups:
        matrix = np.asarray(candidates)
        total += matrix[np.argmax(matrix @ search_weights)]
    return total


def td_targets(rewards, successors, target_weights, gamma):
    # Terminal rows have an all-zero successor feature vector.
    return rewards + gamma * (successors @ target_weights)


def independent_successor(active, boxes, assigned, frame, kalman, next_detections, joint=False):
    from yolox.tracker.basetrack import BaseTrack

    branch_active, branch_boxes = deepcopy((active, boxes))
    id_counter = BaseTrack._count
    try:
        lifecycle = commit_frame(branch_active, branch_boxes, assigned, frame, kalman)
        if next_detections is None:
            return lifecycle, []
        boxes_next, legal_next = prepare_frame(branch_active, next_detections)
        confidence = {det: box.score for det, box in boxes_next.items()}
        _, groups, _ = select_frame_actions(legal_next)
        successor = [
            {'tracks': g['tracks'], 'actions': g['candidates'],
             'features': [candidate_features(g['tracks'], a, legal_next, confidence) for a in g['candidates']]}
            for g in groups
        ]
        if joint:
            from scripts.p4_joint import encode_group, tracking_context

            context = tracking_context(branch_active, frame + 1)
            for group in successor:
                group['joint'] = [encode_group(group['tracks'], action, legal_next, confidence, context)
                                  for action in group['actions']]
        return lifecycle, successor
    finally:
        BaseTrack._count = id_counter


def replay_actions(groups, random):
    yield None, None
    eligible = [g for g in groups if len(g['candidates']) > 1]
    for index in random.permutation(len(eligible))[:ALTERNATIVES_PER_FRAME]:
        group = eligible[index]
        alternatives = [a for a in group['candidates'] if a != group['selected']]
        yield group, alternatives[random.integers(len(alternatives))]


def collect_replay(cache, control, output, joint=False):
    digest = file_sha256(cache)
    manifest = json.loads((control / 'association-summary.json').read_text(encoding='utf-8'))
    if manifest['sequence'] != TRAIN_SEQUENCE or manifest['detections_sha256'] != digest:
        raise ValueError('Replay requires the designated MOT17-05 C1 control and cache')
    length = sequence_info(TRAIN_SEQUENCE).getint('seqLength')
    detections = read_detections(cache, length)
    gt = DATA / 'gt' / TRAIN_SEQUENCE / 'gt' / 'gt.txt'
    truth = load_pedestrian_truth(gt)
    output.mkdir(parents=True, exist_ok=False)
    load_tracker(output)
    from yolox.tracker.basetrack import BaseTrack
    from yolox.tracker.kalman_filter import KalmanFilter

    BaseTrack._count = 0
    active, anchors = {}, {}
    kalman = KalmanFilter()
    random = np.random.default_rng(REPLAY_SEED)
    counters = Counter()
    frame = 0
    with (output / 'transitions.jsonl').open('x', encoding='utf-8') as replay, (
        control / 'association.jsonl'
    ).open(encoding='utf-8') as reference:
        for frame, line in enumerate(reference, 1):
            row = json.loads(line)
            if frame > length or row['frame'] != frame:
                raise ValueError('Control frames are not contiguous')
            team = matched_people(detections[frame], truth[frame])
            boxes, legal = prepare_frame(active, detections[frame])
            assigned, groups, _ = select_frame_actions(legal)
            if assigned != {int(track): det for track, det in row['matches'].items()} or groups != [
                {**g, 'tracks': tuple(g['tracks']), 'candidates': [tuple(a) for a in g['candidates']],
                 'selected': tuple(g['selected'])}
                for g in row['groups']
            ]:
                raise ValueError(f'Replayed C1 choices diverged at frame {frame}')
            confidence = {det: box.score for det, box in boxes.items()}
            if joint:
                from scripts.p4_joint import encode_group, tracking_context

                context = tracking_context(active, frame)
            next_boxes = None if frame == length else detections[frame + 1]
            baseline = None
            for changed_group, alternative in replay_actions(groups, random):
                actions = [alternative if g is changed_group else g['selected'] for g in groups]
                selected = {track: det for g, act in zip(groups, actions, strict=True)
                            for track, det in zip(g['tracks'], act, strict=True)}
                group_scores = [current_quality(g['tracks'], a, legal, anchors, team) / len(g['tracks'])
                                for g, a in zip(groups, actions, strict=True)]
                current = np.zeros(len(FEATURES))
                for g, action in zip(groups, actions, strict=True):
                    current += candidate_features(g['tracks'], action, legal, confidence)
                lifecycle, successor = independent_successor(active, boxes, selected, frame, kalman, next_boxes, joint)
                if changed_group is None:
                    baseline = (lifecycle, successor)
                else:
                    counters['different_successor_inputs'] += successor != baseline[1]
                    counters['different_lifecycles'] += lifecycle != baseline[0]
                record = {'frame': frame, 'changed_tracks': changed_group['tracks'] if changed_group else [],
                          'actions': selected, 'features': current.tolist(), 'group_rewards': group_scores,
                          'reward': sum(group_scores), 'terminal': next_boxes is None,
                          'lifecycle': lifecycle, 'successor': successor}
                if joint:
                    record['joint'] = [encode_group(g['tracks'], act, legal, confidence, context)
                                       for g, act in zip(groups, actions, strict=True)]
                replay.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
                counters['transitions'] += 1
                counters['terminal'] += record['terminal']
            committed = commit_frame(active, boxes, assigned, frame, kalman)
            expected = [row[key] for key in ('births', 'terminations', 'newly_lost', 'reactivations')]
            if [list(pair) for pair in committed[0]] != expected[0] or list(committed[1:]) != expected[1:]:
                raise ValueError(f'Replayed C1 lifecycle diverged at frame {frame}')
            if committed != baseline[0]:
                raise ValueError(f'Independent C1 branch diverged at frame {frame}')
            for track, det in [*assigned.items(), *committed[0]]:
                if det in team:
                    anchors.setdefault(track, team[det])
    if frame != length:
        raise ValueError('Incomplete C1 control')
    metadata = {'sequence': TRAIN_SEQUENCE, 'frames': length, 'detection_cache_sha256': digest,
                'training_gt_sha256': file_sha256(gt), 'seed': REPLAY_SEED,
                'replay_sha256': file_sha256(output / 'transitions.jsonl'), **counters}
    if joint:
        metadata['joint_features'] = True
    (output / 'replay-summary.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata), flush=True)


def train_replay(replay, output, gamma):
    source = replay / 'transitions.jsonl'
    meta = json.loads((replay / 'replay-summary.json').read_text(encoding='utf-8'))
    if meta['sequence'] != TRAIN_SEQUENCE or meta['replay_sha256'] != file_sha256(source):
        raise ValueError('TD replay manifest mismatch')
    with source.open(encoding='utf-8') as file:
        rows = [json.loads(line) for line in file]
    matrix = np.asarray([row['features'] for row in rows])
    rewards = np.asarray([row['reward'] for row in rows])
    successors = [[np.asarray(group['features']) for group in row['successor']] for row in rows]
    terminal = np.asarray([row['terminal'] for row in rows])
    if len(rows) != meta['transitions'] or any(groups for groups, done in zip(successors, terminal) if done):
        raise ValueError('Inconsistent replay or terminal transition')
    online = target = np.zeros(len(FEATURES))
    history = []
    for update in range(1, UPDATE_COUNT + 1):
        best = np.asarray([next_frame_features(groups, online, len(FEATURES)) for groups in successors])
        targets = td_targets(rewards, best, target, gamma)
        if not np.array_equal(targets[terminal], rewards[terminal]):
            raise ValueError('A terminal target incorrectly included another frame')
        if gamma == 0 and not np.array_equal(targets, rewards):
            raise ValueError('Gamma-zero target differs from immediate reward')
        weights = fit_immediate_rewards(matrix, targets)
        if not np.isfinite(weights).all():
            raise RuntimeError(f'TD weights became non-finite at update {update}')
        history.append({'update': update, 'target_min': float(targets.min()), 'target_max': float(targets.max()),
                        'target_std': float(targets.std()), 'weight_update_l2': float(np.linalg.norm(weights - online)),
                        'fit_rmse': float(np.sqrt(np.mean((matrix @ weights - targets) ** 2)))})
        target, online = online, weights
    record = {'objective': 'frame_td', 'gamma': gamma, 'sequence': TRAIN_SEQUENCE, 'feature_names': FEATURES,
              'weights': online.tolist(), 'regularization': REGULARIZATION, 'replay': meta, 'history': history}
    with output.open('x', encoding='utf-8') as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'gamma': gamma, 'weights': record['weights'], 'final_update': history[-1]}), flush=True)


def load_td_ranker(path):
    record = json.loads(path.read_text(encoding='utf-8'))
    weights = np.asarray(record['weights'])
    if (record['objective'] != 'frame_td' or record['gamma'] not in GAMMAS or
            record['sequence'] != TRAIN_SEQUENCE or tuple(record['feature_names']) != FEATURES or
            weights.shape != (len(FEATURES),) or not np.isfinite(weights).all()):
        raise ValueError('Invalid frame TD model')

    def score(group, action, legal, confidences):
        return float(np.asarray(candidate_features(group, action, legal, confidences)) @ weights)

    method = f"P4-TD-g{record['gamma']:.2f}"
    return score, method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect = commands.add_parser('collect')
    collect.add_argument('cache', type=Path)
    collect.add_argument('control', type=Path)
    collect.add_argument('output', type=Path)
    collect.add_argument('--joint', action='store_true')
    fit = commands.add_parser('fit')
    fit.add_argument('replay', type=Path)
    fit.add_argument('output', type=Path)
    fit.add_argument('--gamma', required=True, type=float, choices=GAMMAS)
    args = parser.parse_args()
    if args.command == 'collect':
        collect_replay(args.cache, args.control, args.output, joint=args.joint)
    else:
        train_replay(args.replay, args.output, args.gamma)


if __name__ == '__main__':
    main()
