"""Joint candidate value with individual, mixed, pair, and birth terms."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.detect_mot17_dev import file_sha256
from scripts.p4_supervision import REGULARIZATION, TRAIN_SEQUENCE
from scripts.score_p3_c1 import OPTIONS


INPUT_SIZES = (12, 7, 2)
HIDDEN = 12
LEARNING_RATE = 0.003


def tracking_context(active, frame):
    from yolox.tracker.basetrack import TrackState

    return {track_id: (track.is_activated, track.state == TrackState.Lost, frame - track.end_frame)
            for track_id, track in active.items()}


def encode_group(group, action, legal, confidence, state):
    taken = {detection for detection in action if detection is not None}
    resources = {detection for track in group for detection in legal[track]}
    individual, pair, birth = [], [], []
    for track, det in zip(group, action, strict=True):
        confirmed, lost, age = state[track]
        choices = legal[track]
        matched = det is not None
        surviving = matched or (confirmed and age <= OPTIONS['lost_buffer_frames'])
        shared = [other for other in group if other != track and choices.keys() & legal[other].keys()]
        neighbor_overlap = max((max(legal[other].values(), default=0) for other in shared), default=0)
        individual.append([float(matched), choices.get(det, 0), confidence.get(det, 0),
                           float(confirmed), float(lost), min(age, 30) / 30,
                           float(surviving), float(surviving and not matched),
                           max(choices.values(), default=0), neighbor_overlap,
                           len(shared) / len(group), len(choices) / max(1, len(resources))])
    for index, (left, left_det) in enumerate(zip(group, action, strict=True)):
        for right, right_det in zip(group[index + 1:], action[index + 1:], strict=True):
            shared = legal[left].keys() & legal[right].keys()
            if shared:
                iou_left, iou_right = legal[left].get(left_det, 0), legal[right].get(right_det, 0)
                pair.append([float(left_det is not None) + float(right_det is not None),
                             iou_left + iou_right, abs(iou_left - iou_right),
                             legal[left].get(right_det, 0) + legal[right].get(left_det, 0),
                             confidence.get(left_det, 0) + confidence.get(right_det, 0),
                             max(confidence[d] for d in shared),
                             len(shared) / len(resources)])
    for det in sorted(resources - taken):
        if confidence[det] >= OPTIONS['birth_confidence']:
            birth.append([confidence[det], max(legal[track].get(det, 0) for track in group)])
    return individual, pair, birth


def pack_candidates(candidates):
    arrays = []
    for part, size in enumerate(INPUT_SIZES):
        features, owners = [], []
        for index, sample in enumerate(candidates):
            features.extend(sample[part])
            owners.extend([index] * len(sample[part]))
        arrays.append((torch.tensor(features, dtype=torch.float32).reshape(-1, size),
                       torch.tensor(owners, dtype=torch.long)))
    return len(candidates), arrays


def pool(values, owners, count):
    return values.new_zeros((count, values.shape[1])).index_add_(0, owners, values)


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, outputs))


class JointValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.track = mlp(INPUT_SIZES[0], HIDDEN + 1)
        self.mix = mlp(HIDDEN + 1, 1)
        self.pair = mlp(INPUT_SIZES[1], 1)
        self.birth = mlp(INPUT_SIZES[2], 1)

    def forward(self, packed):
        count, ((tracks, tracks_owner), (pairs, pairs_owner), (births, births_owner)) = packed
        track_count = torch.bincount(tracks_owner, minlength=count).clamp_min(1).unsqueeze(1)
        pooled = pool(self.track(tracks), tracks_owner, count) / track_count
        paired = pool(self.pair(pairs), pairs_owner, count) / track_count
        unborn = pool(self.birth(births), births_owner, count) / track_count
        return (pooled[:, :1] + self.mix(pooled) + paired + unborn).flatten()


def pack_replay(rows):
    current, owners, next_candidates, next_sizes, next_owners = [], [], [], [], []
    for row_index, row in enumerate(rows):
        current.extend(row['joint'])
        owners.extend([row_index] * len(row['joint']))
        for group in row['successor']:
            next_candidates.extend(group['joint'])
            next_sizes.append(len(group['joint']))
            next_owners.append(row_index)
    starts = np.cumsum(next_sizes, dtype=int) - next_sizes
    return pack_candidates(current), torch.tensor(owners), pack_candidates(next_candidates), (
        starts, next_sizes, next_owners
    )


def select_targets(online_values, target_values, layout, count):
    starts, sizes, owners = layout
    selected = [start + np.argmax(online_values[start:start + size])
                for start, size in zip(starts, sizes, strict=True)]
    return torch.from_numpy(np.bincount(owners, weights=target_values[selected], minlength=count)).float()


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit_joint(replay, output, gamma):
    from scripts.p4_td import REPLAY_SEED, UPDATE_COUNT

    configure_cpu()
    torch.manual_seed(REPLAY_SEED)
    source = replay / 'transitions.jsonl'
    meta = json.loads((replay / 'replay-summary.json').read_bytes())
    if not meta.get('joint_features') or meta['sequence'] != TRAIN_SEQUENCE or meta['replay_sha256'] != file_sha256(source):
        raise ValueError('Expected verified joint replay')
    with source.open(encoding='utf-8') as file:
        rows = [json.loads(line) for line in file]
    if len(rows) != meta['transitions'] or any(row['successor'] for row in rows if row['terminal']):
        raise ValueError('Joint replay is incomplete or contains a terminal successor')
    current, owners, successors, layout = pack_replay(rows)
    rewards = torch.tensor([row['reward'] for row in rows], dtype=torch.float32)
    terminal = torch.tensor([row['terminal'] for row in rows])
    online = JointValue()
    target = deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    history = []
    for update in range(1, UPDATE_COUNT + 1):
        with torch.no_grad():
            future = select_targets(online(successors).numpy(), target(successors).numpy(), layout, len(rows))
            targets = rewards + gamma * future
            if not torch.equal(targets[terminal], rewards[terminal]) or (gamma == 0 and not torch.equal(targets, rewards)):
                raise ValueError('Incorrect terminal or gamma-zero target')
            target.load_state_dict(online.state_dict())
        predicted = rewards.new_zeros(len(rows)).index_add_(0, owners, online(current))
        loss = (predicted - targets).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at update {update}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {name: float(sum(p.grad.square().sum() for p in module.parameters()).sqrt())
                     for name, module in online.named_children()}
        optimizer.step()
        history.append({'update': update, 'rmse': float(loss.detach().sqrt()), 'target_min': float(targets.min()),
                        'target_max': float(targets.max()), 'target_std': float(targets.std(unbiased=False)),
                        'gradient_l2': gradients})
    record = {'objective': 'joint_td', 'gamma': gamma, 'sequence': TRAIN_SEQUENCE, 'replay': meta,
              'regularization': REGULARIZATION, 'learning_rate': LEARNING_RATE,
              'parameter_count': sum(p.numel() for p in online.parameters()), 'history': history,
              'state': {name: parameter.tolist() for name, parameter in online.state_dict().items()}}
    with output.open('x', encoding='utf-8') as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({key: record[key] for key in ('gamma', 'parameter_count')} |
                     {'initial': history[0], 'final': history[-1]}), flush=True)


class JointRanker:
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def score_candidates(self, group, proposals, legal, confidence, state):
        encoded = [encode_group(group, action, legal, confidence, state) for action in proposals]
        return self.model(pack_candidates(encoded)).tolist()


def load_joint_ranker(path):
    from scripts.p4_td import GAMMAS

    configure_cpu()
    record = json.loads(path.read_bytes())
    if record['objective'] != 'joint_td' or record['gamma'] not in GAMMAS or record['sequence'] != TRAIN_SEQUENCE:
        raise ValueError('Invalid joint model')
    model = JointValue()
    weights = {name: torch.tensor(values) for name, values in record['state'].items()}
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('Non-finite joint model')
    model.load_state_dict(weights)
    return JointRanker(model.eval()), f"P4-joint-g{record['gamma']:.2f}"


if __name__ == '__main__':
    from scripts.p4_td import GAMMAS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--gamma', required=True, choices=GAMMAS, type=float)
    args = parser.parse_args()
    fit_joint(args.replay, args.output, args.gamma)
