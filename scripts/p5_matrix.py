"""Run and record the P5 DanceTrack mechanism matrix in dependency order."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from scripts.detect_mot17_dev import file_sha256
from scripts.p5_dance_data import official_names, sequences, verify_cache
from scripts.p5_dance_eval import GATES
from scripts.p5_train import EPOCHS


class Matrix:
    def __init__(self, output, cache, workers):
        self.output = output.resolve()
        self.cache = cache.resolve()
        self.workers = workers
        self.logs = self.output / 'logs'
        self.logs.mkdir(parents=True, exist_ok=True)
        self.split = official_names('val')

    def command(self, key, module, *args):
        log = self.logs / f'{key}.log'
        with log.open('ab') as file:
            cmd = [sys.executable, '-u', '-m', f'scripts.{module}', *map(str, args)]
            file.write((subprocess.list2cmdline(cmd) + '\n').encode())
            file.flush()
            subprocess.run(cmd, check=True, stdout=file, stderr=subprocess.STDOUT)

    def parallel(self, function, tasks):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(function, tasks))

    def score(self, key, gate, method='C1', checkpoint=None, weight=0.10, budget=16):
        run = self.output / 'eval' / key
        before = run / 'input-manifest.json'
        results = list((run / 'trackeval').glob('*/pedestrian_summary.txt'))
        if not before.is_file() or len(results) != 1:
            if run.exists():
                raise ValueError(f'Incomplete run requires inspection before retry: {run}')
            arg = ['--checkpoint', checkpoint] if checkpoint else []
            self.command(key, 'p5_dance_eval', run, self.cache, '--method', method, '--hand-weight', weight,
                         '--gate', gate, '--k', budget, *arg, *(['--timings'] if method != 'ByteTrack' else []))
            results = list((run / 'trackeval').glob('*/pedestrian_summary.txt'))
        manifest = json.loads(before.read_bytes())
        if sorted(manifest['coverage']) != self.split or len(results) != 1:
            raise ValueError(f'Missing full official validation aggregate for {key}')
        if checkpoint and manifest.get('checkpoint_sha256') != file_sha256(checkpoint):
            raise ValueError(f'Checkpoint mismatch in {key}')
        names, values = results[0].read_text().splitlines()
        metrics = dict(zip(names.split(), map(float, values.split()), strict=True))
        average = float('inf')
        if (run / 'association-timings.json').is_file():
            timing = json.loads((run / 'association-timings.json').read_bytes())
            average = sum(row[0] for frames in timing.values() for row in frames) / sum(map(len, timing.values())) / 1e6
        row = {'key': key, 'metrics': metrics, 'association_ms': average if average != float('inf') else None,
               'run': str(run), 'gate': gate, 'candidate_limit': budget, 'method': manifest['method']}
        print(json.dumps({'completed_eval': key, 'HOTA': metrics['HOTA']}), flush=True)
        return row

    @staticmethod
    def pick(rows):
        return min(rows, key=lambda r: (-r['metrics']['HOTA'], r['association_ms'], r.get('epoch', 0), r['key']))

    def validate_caches(self, split):
        for folder, info in sequences(split):
            verify_cache(self.cache / split / f'{folder.name}.txt', split, folder.name, info.getint('seqLength'))

    def baseline(self):
        self.validate_caches('val')
        raw = self.parallel(lambda gate: self.score(f'c1-{gate}', gate), list(GATES))
        selection = self.pick(raw)
        byte = self.score('c0-bytetrack', selection['gate'], method='ByteTrack')
        record = {'gate_selection': selection, 'c1_search': raw, 'c0': byte}
        self.save('gate-selection.json', record)
        return record

    def collect(self, gate):
        self.validate_caches('train')
        names = official_names('train')
        replay, packed = self.output / 'replay', self.output / 'packed'

        def sequence(name):
            self.command(f'collect-{name}', 'p5_replay', replay, self.cache, '--sequence', name, '--gate', gate)
            self.command(f'pack-{name}', 'p5_train', 'pack', replay, packed, '--sequence', name)
            print(f'Prepared replay: {name}', flush=True)

        self.parallel(sequence, names)

    def train_evaluate(self, gate, spec):
        variant, objective, gamma, seed = spec
        key = f'{variant}-{objective}-{gamma:.2f}-s{seed}'
        target = self.output / 'models' / key
        if not (target / f'epoch-{EPOCHS}.json').is_file():
            if target.exists():
                raise ValueError(f'Incomplete training run requires inspection before retry: {target}')
            self.command(f'train-{key}', 'p5_train', 'fit', self.output / 'packed', target,
                         '--variant', variant, '--objective', objective, '--gamma', gamma, '--seed', seed)
        choices = []
        for epoch in range(1, EPOCHS + 1):
            checkpoint = target / f'epoch-{epoch}.json'
            choices.append({**self.score(f'{key}-ep{epoch}', gate, checkpoint=checkpoint),
                            'epoch': epoch, 'checkpoint': str(checkpoint), 'variant': variant,
                            'objective': objective, 'gamma': gamma, 'seed': seed})
        winner = self.pick(choices)
        self.save(f'selections/{key}.json', {'winner': winner, 'checkpoints': choices})
        return winner

    def training(self, gate):
        modes = [('supervised', 0.0), ('td', 0.0), ('td', 0.9), ('td', 0.99)]
        e4 = self.parallel(lambda spec: self.train_evaluate(gate, spec),
                           [('C3', objective, gamma, seed) for seed in range(3) for objective, gamma in modes])
        winner = self.pick([row for row in e4 if row['seed'] == 0 and row['gamma'] != 0 and row['objective'] == 'td'])
        self.save('e4-selection.json', {'e4': e4, 'selection': winner})
        variants = ['C2', 'C2-wide', 'no-neighborhood', 'sum-mix', 'no-pair']
        e23 = self.parallel(lambda spec: self.train_evaluate(gate, spec),
                            [(variant, 'td', winner['gamma'], seed) for seed in range(3) for variant in variants])
        self.save('tables-e2-e4.json', {'e4': e4, 'e23': e23, 'selection': winner})
        return winner

    def search(self, gate, selection):
        checkpoint = Path(selection['checkpoint'])
        weights = [0.05, 0.10, 0.20]
        hand_search = self.parallel(
            lambda weight: {**self.score(f'e5-hand-w{weight:.2f}', gate, method='hand', weight=weight),
                            'hand_weight': weight}, weights
        )
        hand_weight = self.pick(hand_search)['hand_weight']

        def score_case(case):
            method, k = case
            if k == 16 and method == 'C1':
                return self.score(f'c1-{gate}', gate)
            if k == 16 and method == 'hand':
                return self.score(f'e5-hand-w{hand_weight:.2f}', gate, method='hand', weight=hand_weight)
            if k == 16 and method == 'learned':
                return self.score(selection['key'], gate, checkpoint=checkpoint)
            return self.score(f'e5-{method}-k{k}', gate, method='hand' if method == 'hand' else 'C1',
                              checkpoint=checkpoint if method == 'learned' else None, weight=hand_weight, budget=k)

        rows = self.parallel(score_case, [(m, k) for m in ('C1', 'hand', 'learned') for k in (1, 4, 8, 16, 32)])
        self.save('table-e5-search.json', {'hand_weight': hand_weight, 'hand_search': hand_search, 'closed_loop': rows})
        return hand_weight

    def save(self, name, record):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix('.tmp')
        staged.write_text(json.dumps(record, allow_nan=False, indent=2) + '\n', encoding='utf-8')
        staged.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stage', choices=('baseline', 'collect', 'train', 'search', 'all'), default='all')
    args = parser.parse_args()
    job = Matrix(args.output, args.cache, args.workers)
    if args.stage in ('baseline', 'all'):
        baseline = job.baseline()
    else:
        baseline = json.loads((job.output / 'gate-selection.json').read_bytes())
    gate = baseline['gate_selection']['gate']
    if args.stage in ('collect', 'all'):
        job.collect(gate)
    if args.stage in ('train', 'all'):
        selected = job.training(gate)
    if args.stage in ('search', 'all'):
        if args.stage == 'search':
            selected = json.loads((job.output / 'tables-e2-e4.json').read_bytes())['selection']
        job.search(gate, selected)


if __name__ == '__main__':
    main()
