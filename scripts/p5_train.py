"""Train and load CGHTrack mechanism controls on frozen, sharded replay."""

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.detect_mot17_dev import file_sha256
from scripts.p4_joint import HIDDEN, INPUT_SIZES, JointRanker, JointValue, configure_cpu, encode_group, pack_candidates, pack_replay, pool
from scripts.p5_dance_data import official_names
from scripts.p5_mot_detector import TRAIN as MOT_TRAIN
from scripts.score_p3_c1 import OPTIONS


LOCAL_NEIGHBOR_COLUMNS = (9, 10, 11)
VARIANTS = ('C3', 'C2', 'C2-wide', 'no-neighborhood', 'sum-mix', 'no-pair')
BATCH = 128
EPOCHS = 3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
TARGET_INTERVAL = 32
DISCOUNTS = (0.0, 0.9, 0.99)


class AblatedJointValue(JointValue):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        if variant in ('C2', 'C2-wide', 'sum-mix'):
            width = 44 if variant == 'C2-wide' else HIDDEN
            self.track = nn.Sequential(nn.Linear(INPUT_SIZES[0], width), nn.Tanh(), nn.Linear(width, 1))
            self.mix = None
        if variant in ('C2', 'C2-wide', 'no-pair'):
            self.pair = None

    def forward(self, packed):
        count, ((tracks, tracks_owner), (pairs, pairs_owner), (births, births_owner)) = packed
        if self.variant == 'no-neighborhood':
            tracks = tracks.clone()
            tracks[:, LOCAL_NEIGHBOR_COLUMNS] = 0
        track_count = torch.bincount(tracks_owner, minlength=count).clamp_min(1).unsqueeze(1)
        pooled = pool(self.track(tracks), tracks_owner, count) / track_count
        score = pooled[:, :1] + pool(self.birth(births), births_owner, count) / track_count
        if self.mix is not None:
            score = score + self.mix(pooled)
        if self.pair is not None:
            score = score + pool(self.pair(pairs), pairs_owner, count) / track_count
        return score.flatten()


def make_model(variant):
    return JointValue() if variant == 'C3' else AblatedJointValue(variant)


class P5Ranker(JointRanker):
    @torch.inference_mode()
    def score_groups(self, groups, legal, confidence, state):
        encoded = [encode_group(group['tracks'], action, legal, confidence, state)
                   for group in groups for action in group['candidates']]
        values = self.model(pack_candidates(encoded)).tolist()
        cursor = 0
        for group in groups:
            end = cursor + len(group['candidates'])
            yield values[cursor:end]
            cursor = end


def prepare_batch(rows):
    current, current_owners, successors, (_, sizes, owners) = pack_replay(rows)
    pair_feats = [candidate for row in rows for candidate in row['preference']]
    group_owners = torch.tensor(owners, dtype=torch.long)
    candidate_sizes = torch.tensor(sizes, dtype=torch.long)
    candidate_groups = torch.repeat_interleave(torch.arange(len(sizes)), candidate_sizes)
    return {'current': current, 'owners': current_owners.long(), 'successors': successors,
            'candidate_groups': candidate_groups, 'next_group_owners': group_owners,
            'rewards': torch.tensor([row['reward'] for row in rows], dtype=torch.float32),
            'terminal': torch.tensor([row['terminal'] for row in rows], dtype=torch.bool),
            'pairs': pack_candidates(pair_feats)}


def training_names(dataset):
    return MOT_TRAIN if dataset == 'MOT17' else official_names('train')


def prepare_replay(source, output, only=None, dataset='DanceTrack'):
    configure_cpu()
    output.mkdir(parents=True, exist_ok=True)
    names = training_names(dataset)
    if only is not None:
        names = [only] if only in names else []
    if not names:
        raise ValueError('Missing official training sequence')
    for name in names:
        src = source / f'{name}.jsonl.gz'
        meta = json.loads((source / f'{name}.json').read_bytes())
        dest = output / f'{name}.pt'
        manifest = dest.with_suffix('.json')
        if (meta['dataset'] != dataset or meta['split'] != 'train' or meta['sequence'] != name or
                meta['sha256'] != file_sha256(src)):
            raise ValueError(f'Unexpected replay for {name}')
        if dest.is_file() and manifest.is_file():
            previous = json.loads(manifest.read_bytes())
            if (previous['source_sha256'] == meta['sha256'] and previous['sha256'] == file_sha256(dest)
                    and previous['gate_options'] == meta['gate_options']):
                continue
            raise ValueError(f'Unexpected packed replay {name}')
        with gzip.open(src, 'rt', encoding='utf-8') as file:
            rows = [json.loads(line) for line in file]
        if len(rows) != meta['rows'] or any(row['terminal'] and row['successor'] for row in rows):
            raise ValueError(f'Replay offsets are invalid for {name}')
        np.random.default_rng(0).shuffle(rows)
        blocks = [prepare_batch(rows[i:i+BATCH]) for i in range(0, len(rows), BATCH)]
        staged = dest.with_suffix('.tmp')
        torch.save(blocks, staged)
        staged.replace(dest)
        manifest.write_text(json.dumps({'dataset': dataset, 'sequence': name, 'split': 'train', 'rows': len(rows),
                                        'blocks': len(blocks), 'source_sha256': meta['sha256'],
                                        'gate_options': meta['gate_options'],
                                        'sha256': file_sha256(dest)}, indent=2) + '\n', encoding='utf-8')
        print(f'Prepared {name}: {len(rows)} transitions, {len(blocks)} batches', flush=True)


def next_values(online, target, batch):
    groups = batch['candidate_groups']
    owners = batch['next_group_owners']
    output = batch['rewards'].new_zeros(len(batch['rewards']))
    if not groups.numel():
        return output
    values = online(batch['successors'])
    best = values.new_full((len(owners),), -torch.inf).scatter_reduce_(0, groups, values, reduce='amax')
    indices = torch.arange(len(values))
    ties = torch.where(values == best[groups], indices, len(values))
    winners = indices.new_full((len(owners),), len(values)).scatter_reduce_(0, groups, ties, reduce='amin')
    return output.index_add_(0, owners, target(batch['successors'])[winners])


def training_loss(online, target, batch, objective, gamma):
    if objective == 'supervised':
        values = online(batch['pairs'])
        loss = F.softplus(values[1::2] - values[::2]).sum() / len(batch['rewards'])
        return loss, batch['rewards']
    rewards = batch['rewards']
    with torch.no_grad():
        targets = rewards if gamma == 0 else rewards + gamma * next_values(online, target, batch)
        if not torch.equal(targets[batch['terminal']], rewards[batch['terminal']]):
            raise RuntimeError('Terminal value included a future frame')
    predicted = torch.zeros_like(rewards).index_add_(0, batch['owners'], online(batch['current']))
    return (predicted - targets).square().mean(), targets


def fit(replay, output, variant, seed, objective, gamma, dataset='DanceTrack'):
    configure_cpu()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if objective == 'supervised' and gamma != 0:
        raise ValueError('Current-frame supervision requires gamma zero')
    shards = {name: replay / f'{name}.pt' for name in training_names(dataset)}
    manifests = {name: json.loads(path.with_suffix('.json').read_bytes()) for name, path in shards.items()}
    for name, path in shards.items():
        meta = manifests[name]
        if (meta['dataset'] != dataset or meta['split'] != 'train' or meta['sequence'] != name or
                file_sha256(path) != meta['sha256']):
            raise ValueError(f'Inconsistent packed training shard: {path}')
    gate_options = next(iter(manifests.values()))['gate_options']
    if any(meta['gate_options'] != gate_options for meta in manifests.values()):
        raise ValueError('Training shards use different candidate gates')
    online = make_model(variant)
    target = deepcopy(online).eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    output.mkdir(parents=True, exist_ok=False)
    completed = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        losses, target_sum, target_squares, targets_count = [], 0.0, 0.0, 0
        for name in rng.permutation(list(shards)):
            blocks = torch.load(shards[name], map_location='cpu', weights_only=True)
            for index in rng.permutation(len(blocks)):
                batch = blocks[index]
                loss, targets = training_loss(online, target, batch, objective, gamma)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite {variant} loss in {name}, epoch {epoch}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                completed += 1
                if completed % TARGET_INTERVAL == 0:
                    target.load_state_dict(online.state_dict())
                losses.append(float(loss.detach()))
                target_sum += float(targets.sum())
                target_squares += float(targets.square().sum())
                targets_count += targets.numel()
        deviation = np.sqrt(max(0, target_squares / targets_count - (target_sum / targets_count) ** 2))
        history.append({'epoch': epoch, 'steps': completed, 'loss': float(np.mean(losses)),
                        'target_std': float(deviation)})
        record = {'dataset': dataset, 'split': 'train', 'seed': seed, 'variant': variant,
                  'objective': objective, 'gamma': gamma, 'history': history,
                  'gate_options': gate_options,
                  'parameter_count': sum(p.numel() for p in online.parameters()), 'replay': manifests,
                  'state': {name: value.tolist() for name, value in online.state_dict().items()}}
        path = output / f'epoch-{epoch}.json'
        path.write_text(json.dumps(record, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'method': variant, 'objective': objective, 'gamma': gamma, 'seed': seed, **history[-1]}),
              flush=True)


def load_ranker(path):
    configure_cpu()
    record = json.loads(path.read_bytes())
    if record['gate_options'] != OPTIONS:
        raise ValueError('Checkpoint requires its original candidate threshold configuration')
    variant, seed, objective, gamma = (record[key] for key in ('variant', 'seed', 'objective', 'gamma'))
    model = make_model(variant)
    model.load_state_dict({name: torch.tensor(values) for name, values in record['state'].items()})
    tag = 'supervised' if objective == 'supervised' else f'g{gamma:.2f}'
    return P5Ranker(model.eval()), f'{variant}-{tag}-s{seed}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pack = commands.add_parser('pack')
    pack.add_argument('source', type=Path)
    pack.add_argument('output', type=Path)
    pack.add_argument('--sequence')
    pack.add_argument('--dataset', choices=('DanceTrack', 'MOT17'), default='DanceTrack')
    train = commands.add_parser('fit')
    train.add_argument('source', type=Path)
    train.add_argument('output', type=Path)
    train.add_argument('--variant', choices=VARIANTS, required=True)
    train.add_argument('--seed', choices=(0, 1, 2), type=int, required=True)
    train.add_argument('--objective', choices=('td', 'supervised'), default='td')
    train.add_argument('--gamma', choices=DISCOUNTS, type=float, required=True)
    train.add_argument('--dataset', choices=('DanceTrack', 'MOT17'), default='DanceTrack')
    args = parser.parse_args()
    if args.command == 'pack':
        prepare_replay(args.source, args.output, args.sequence, args.dataset)
    else:
        fit(args.source, args.output, args.variant, args.seed, args.objective, args.gamma, args.dataset)
