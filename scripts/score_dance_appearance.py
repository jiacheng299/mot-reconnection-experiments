"""Run the appearance baselines on the verified DanceTrack val cache."""

import argparse
import json
from pathlib import Path

import trackeval

from detect_mot17_dev import ROOT, file_sha256
from run_dance_dev import SPLIT, sequences
from score_appearance_mot17 import verify_source, write_tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=('BoT-SORT', 'TrackTrack'))
    parser.add_argument('run', type=Path)
    parser.add_argument('--sequence', help='optional one-sequence engineering check')
    args = parser.parse_args()
    verify_source(args.method)
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    config_dir = run / 'configs'
    config_dir.mkdir(exist_ok=True)
    seq_info = {}
    for folder, info in sequences():
        name = folder.name
        if args.sequence and name != args.sequence:
            continue
        length = info.getint('seqLength')
        seq_info[name] = length
        cache = ROOT / 'runs' / 'p2-dance-001' / 'detections' / f'{name}.txt'
        record = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
        digest = file_sha256(cache)
        if record['frames'] != length or digest != record['detections_sha256']:
            raise ValueError(f'Invalid shared cache for {name}')
        settings = config_dir / f'{name}.json'
        output = run / 'trackers' / args.method / 'data' / f'{name}.txt'
        if settings.is_file() and output.is_file():
            saved = json.loads(settings.read_text(encoding='utf-8'))
            if saved['detections_sha256'] != digest or saved['predictions_sha256'] != file_sha256(output):
                raise ValueError(f'Incomplete or changed result for {name}')
            continue
        images = sorted((folder / info['imDir']).glob(f'*{info["imExt"]}'), key=lambda p: int(p.stem))
        if [int(p.stem) for p in images] != list(range(1, length + 1)):
            raise ValueError(f'Incomplete or misaligned images for {name}')
        print(f'Using {name} shared cache SHA-256 {digest}', flush=True)
        config, _ = write_tracks(run, args.method, info, cache, images)
        config['predictions_sha256'] = file_sha256(output)
        settings.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    if not seq_info:
        raise ValueError('No sequences selected')

    print(f'Evaluation coverage: {len(seq_info)}/25 official val sequences', flush=True)
    evaluator = trackeval.Evaluator({'USE_PARALLEL': False, 'PLOT_CURVES': False})
    dataset = trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER': str(SPLIT), 'TRACKERS_FOLDER': str(run / 'trackers'),
        'OUTPUT_FOLDER': str(run / 'trackeval'), 'TRACKERS_TO_EVAL': [args.method],
        'BENCHMARK': 'DanceTrack', 'SPLIT_TO_EVAL': 'val', 'SKIP_SPLIT_FOL': True,
        'SEQ_INFO': seq_info, 'DO_PREPROC': True,
    })
    evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])


if __name__ == '__main__':
    main()
