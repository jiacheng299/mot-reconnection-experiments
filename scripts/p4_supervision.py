"""Fit P4 current-frame ranking or immediate team reward models."""

from collections import defaultdict
import csv
import json
from math import fsum
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit

from scripts.detect_mot17_dev import DATA, file_sha256, sequence_info
from scripts.run_p1 import read_detections


TRAIN_SEQUENCE = 'MOT17-05-FRCNN'
FEATURES = ('match_fraction', 'mean_iou', 'mean_confidence', 'mean_iou_times_confidence')
REGULARIZATION = 0.01


def candidate_features(group, action, legal, confidence):
    matches = [(legal[track][detection], confidence[detection])
               for track, detection in zip(group, action, strict=True) if detection is not None]
    size = len(group)
    return (len(matches) / size, fsum(iou for iou, _ in matches) / size,
            fsum(score for _, score in matches) / size,
            fsum(iou * score for iou, score in matches) / size)


def matched_people(detections, people):
    """Create one-to-one training targets at IoU >= 0.5."""
    if not detections or not people:
        return {}
    identities = sorted(people)
    boxes = np.asarray(detections)[:, :4]
    truth = np.asarray([people[identity] for identity in identities])
    overlap = np.maximum(0, np.minimum(boxes[:, None, 2:], truth[None, :, 2:]) -
                         np.maximum(boxes[:, None, :2], truth[None, :, :2])).prod(axis=2)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)[:, None]
    truth_area = (truth[:, 2:] - truth[:, :2]).prod(axis=1)[None, :]
    iou = overlap / (area + truth_area - overlap)
    eligible = (iou >= 0.5) & (np.asarray(detections)[:, 4, None] >= 0.1)
    score = np.where(eligible, 1 + iou / (1 + min(iou.shape)), 0)
    rows, cols = linear_sum_assignment(score, maximize=True)
    return {int(row) + 1: identities[col] for row, col in zip(rows, cols) if score[row, col] > 0}


def current_quality(group, action, legal, anchors, people):
    """Reward preserving an anchored identity and penalize avoidable misses."""
    quality = 0
    for track, detection in zip(group, action, strict=True):
        identity = anchors.get(track)
        if identity is None:
            continue
        if detection is not None:
            quality += 1 if people.get(detection) == identity else -1
        elif any(people.get(candidate) == identity for candidate in legal[track]):
            quality -= 1
    return quality


def fit_immediate_rewards(features, rewards):
    matrix = np.asarray(features, dtype=float)
    targets = np.asarray(rewards, dtype=float)
    count, dimensions = matrix.shape
    # Solve the normal equations for mean squared error plus L2 penalty.
    return np.linalg.solve(matrix.T @ matrix / count + REGULARIZATION * np.eye(dimensions),
                           matrix.T @ targets / count)


def load_pedestrian_truth(path):
    people = defaultdict(dict)
    with path.open(newline='') as file:
        for row in csv.reader(file):
            if int(row[6]) != 1 or int(row[7]) != 1:
                continue
            frame, identity = map(int, row[:2])
            x, y, width, height = map(float, row[2:6])
            people[frame][identity] = (x, y, x + width, y + height)
    return people


def train_ranker(run, cache, output, immediate=False):
    length = sequence_info(TRAIN_SEQUENCE).getint('seqLength')
    metadata = json.loads((run / 'association-summary.json').read_text(encoding='utf-8'))
    digest = file_sha256(cache)
    if metadata['sequence'] != TRAIN_SEQUENCE or metadata['detections_sha256'] != digest:
        raise ValueError('The training rollout must use the designated MOT17-05 cache')
    detections = read_detections(cache, length)
    gt = DATA / 'gt' / TRAIN_SEQUENCE / 'gt' / 'gt.txt'
    people = load_pedestrian_truth(gt)

    anchors = {}
    differences = []
    reward_features, rewards = [], []
    example = None
    supervised_groups = 0
    expected = 0
    with (run / 'association.jsonl').open(encoding='utf-8') as file:
        for expected, line in enumerate(file, start=1):
            row = json.loads(line)
            if row['frame'] != expected:
                raise ValueError('Training frames must be ordered and contiguous')
            boxes = detections.get(expected, [])
            confidences = {i: box[4] for i, box in enumerate(boxes, start=1)}
            assigned_people = matched_people(boxes, people[expected])
            legal = {int(t): {int(d): score for d, score in values.items()}
                     for t, values in row['legal_scores'].items()}
            for group in row['groups']:
                tracks, actions = group['tracks'], group['candidates']
                if not any(track in anchors for track in tracks):
                    continue
                labels = [current_quality(tracks, act, legal, anchors, assigned_people) for act in actions]
                if immediate:
                    supervised_groups += 1
                    reward_features.extend(candidate_features(tracks, act, legal, confidences) for act in actions)
                    rewards.extend(label / len(tracks) for label in labels)
                    if example is None and len(tracks) > 1 and min(labels) < 0 < max(labels):
                        example = {'frame': expected, 'tracks': tracks, 'actions': actions,
                                   'track_identity_anchors': [anchors.get(track) for track in tracks],
                                   'detection_identities': {det: assigned_people.get(det)
                                                            for act in actions for det in act if det is not None},
                                   'quality_scores': labels}
                    continue
                best = max(labels)
                if min(labels) == best:
                    continue
                supervised_groups += 1
                xs = [np.asarray(candidate_features(tracks, act, legal, confidences)) for act in actions]
                differences.extend(xs[i] - xs[j] for i, label in enumerate(labels) if label == best
                                   for j, worse in enumerate(labels) if worse < best)
            # Update anchors only after this frame's labels. Freeze the first matched identity.
            committed = [(int(t), det) for t, det in row['matches'].items()] + row['births']
            for track, detection in committed:
                if detection in assigned_people:
                    anchors.setdefault(track, assigned_people[detection])
    if expected != length or not (rewards if immediate else differences):
        raise ValueError('Incomplete training sequence or no training samples')
    record = {'sequence': TRAIN_SEQUENCE, 'frames': length, 'detection_cache_sha256': digest,
              'training_gt_sha256': file_sha256(gt), 'feature_names': FEATURES}
    if immediate:
        matrix = np.asarray(reward_features)
        targets = np.asarray(rewards)
        weights = fit_immediate_rewards(matrix, targets)
        if not np.isfinite(weights).all():
            raise RuntimeError('Non-finite immediate reward weights')
        record.update(objective='immediate_team_reward', gamma=0.0, weights=weights.tolist(),
                      supervised_groups=supervised_groups, reward_samples=len(targets),
                      labelled_tracks=len(anchors), regularization=REGULARIZATION,
                      reward_counts={str(float(target)): int((targets == target).sum()) for target in np.unique(targets)},
                      initial_rmse=float(np.sqrt(np.mean(targets ** 2))),
                      final_rmse=float(np.sqrt(np.mean((matrix @ weights - targets) ** 2))),
                      example=example)
    else:
        record.update(fit_pairwise_ranking(differences, supervised_groups, len(anchors)))
    with output.open('x', encoding='utf-8') as file:
        json.dump(record, file, indent=2)
        file.write('\n')
    print(json.dumps(record), flush=True)


def fit_pairwise_ranking(differences, supervised_groups, labelled_tracks):
    matrix = np.asarray(differences)

    def objective(weights):
        margins = matrix @ weights
        loss = np.logaddexp(0, -margins).mean() + 0.5 * REGULARIZATION * (weights @ weights)
        gradient = -(matrix.T @ expit(-margins)) / len(matrix) + REGULARIZATION * weights
        return loss, gradient

    fit = minimize(objective, np.zeros(matrix.shape[1]), jac=True, method='L-BFGS-B')
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f'Current-frame ranking fit failed: {fit.message}')
    return {'weights': fit.x.tolist(), 'supervised_groups': supervised_groups,
            'preference_pairs': len(matrix), 'labelled_tracks': labelled_tracks,
            'regularization': REGULARIZATION, 'iterations': fit.nit,
            'initial_loss': float(np.log(2)), 'final_loss': float(fit.fun),
            'train_pair_accuracy': float((matrix @ fit.x > 0).mean())}


def load_ranker(path, immediate=False):
    record = json.loads(path.read_text(encoding='utf-8'))
    weights = record['weights']
    if (record['sequence'] != TRAIN_SEQUENCE or tuple(record['feature_names']) != FEATURES or
            len(weights) != len(FEATURES) or not np.isfinite(weights).all()):
        raise ValueError('Invalid current-frame ranking checkpoint')
    if immediate:
        if record.get('objective') != 'immediate_team_reward' or record.get('gamma') != 0:
            raise ValueError('Expected a gamma-zero immediate reward checkpoint')
    elif 'objective' in record:
        raise ValueError('Expected a current-frame ranking checkpoint')

    def score(group, action, legal, confidences):
        return fsum(w * f for w, f in zip(weights, candidate_features(group, action, legal, confidences), strict=True))

    return score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--immediate', action='store_true')
    args = parser.parse_args()
    train_ranker(args.rollout, args.cache, args.output, args.immediate)
