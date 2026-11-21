"""Fit and evaluate the three-seed MOT17 E2 confirmation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from scripts.p5_train import EPOCHS


def run(packed, cache, output, gamma, workers, variants, candidate_only=False):
    output.mkdir(parents=True, exist_ok=True)

    def command(key, module, *args):
        log = output / 'logs' / f'{key}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('ab') as file:
            subprocess.run([sys.executable, '-u', '-m', f'scripts.{module}', *map(str, args)],
                           stdout=file, stderr=subprocess.STDOUT, check=True)

    def model(task):
        variant, seed = task
        name = f'{variant}-g{gamma:.2f}-s{seed}'
        models = output / 'models' / name
        if not (models / f'epoch-{EPOCHS}.json').is_file():
            command(f'{name}-train', 'p5_train', 'fit', packed, models, '--variant', variant,
                    '--seed', seed, '--gamma', gamma, '--dataset', 'MOT17')
        scores = []
        for epoch in range(1, EPOCHS + 1):
            checkpoint = models / f'epoch-{epoch}.json'
            key = f'{name}-ep{epoch}'
            evaluation = output / 'eval' / key
            if not (evaluation / 'input-manifest.json').is_file():
                command(key, 'p5_mot_eval', evaluation, cache, '--checkpoint', checkpoint)
            results, = (evaluation / 'trackeval').glob('*/pedestrian_summary.txt')
            manifest = json.loads((evaluation / 'input-manifest.json').read_bytes())
            metrics, values = results.read_text().splitlines()
            scores.append({'epoch': epoch, 'checkpoint': str(checkpoint.resolve()), 'key': key,
                           'association_ms': manifest['association_ms'],
                           'summary': str(results.resolve()),
                           'metrics': dict(zip(metrics.split(), map(float, values.split()), strict=True))})
        winner = min(scores, key=lambda item: (-item['metrics']['HOTA'], item['association_ms'], item['epoch']))
        print(json.dumps({'completed': name, 'HOTA': winner['metrics']['HOTA']}), flush=True)
        return {'variant': variant, 'seed': seed, 'gamma': gamma, 'choices': scores, 'winner': winner}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(model, [(variant, seed) for variant in variants for seed in range(3)]))
    path = output / (f'mot17-e2-candidate-g{gamma:.2f}.json' if candidate_only else 'mot17-e2.json')
    partial = path.with_suffix('.tmp')
    partial.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    partial.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packed', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--gamma', required=True, type=float, choices=(0.9, 0.99))
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--variants', nargs='+', default=('C2', 'C2-wide', 'C3'))
    parser.add_argument('--candidate-only', action='store_true')
    args = parser.parse_args()
    run(args.packed, args.cache, args.output, args.gamma, args.workers, args.variants, args.candidate_only)
