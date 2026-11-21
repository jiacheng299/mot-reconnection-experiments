"""Export and score the locally generated single-sequence CAMELTrack run."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import trackeval

from detect_mot17_dev import ROOT, file_sha256
from run_dance_dev import SPLIT, sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='local TrackLab run containing output.pklz')
    parser.add_argument('sequence')
    parser.add_argument('--min-input-score', type=float, default=0.0)
    args = parser.parse_args()
    run, sequence = args.run.resolve(), args.sequence
    length = next(info.getint('seqLength') for folder, info in sequences() if folder.name == sequence)
    cache = ROOT / 'runs' / 'p2-dance-001' / 'detections' / f'{sequence}.txt'
    record = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
    if file_sha256(cache) != record['detections_sha256']:
        raise ValueError('Shared DanceTrack cache changed')
    expected = defaultdict(list)
    with cache.open(newline='') as file:
        for row in csv.reader(file):
            if float(row[6]) >= args.min_input_score:
                expected[int(row[0])].append(tuple(map(float, row[2:7])))
    print(f'Using {sequence} shared SHA-256 {record["detections_sha256"]}', flush=True)

    # Only read the trusted, locally generated tracker output.
    with ZipFile(run / 'output.pklz') as state:
        names = [name for name in state.namelist() if name.endswith('_image.pkl')]
        if len(names) != 1:
            raise ValueError('Expected exactly one video in the TrackLab state')
        with state.open(names[0]) as file:
            images = pd.read_pickle(file)
        with state.open(names[0].replace('_image.pkl', '.pkl')) as file:
            detections = pd.read_pickle(file)
    frames = {index: int(Path(path).stem) for index, path in images['file_path'].items()}
    if sorted(frames.values()) != list(range(1, length + 1)) or len(detections) != sum(map(len, expected.values())):
        raise ValueError('TrackLab input coverage does not match the shared cache')
    if {Path(path).parent.parent.name for path in images['file_path']} != {sequence}:
        raise ValueError('TrackLab state contains an unexpected sequence')

    rows = []
    for row in detections.itertuples():
        frame = frames[row.image_id]
        coords = (*row.bbox_ltwh, row.bbox_conf)
        if not any(np.allclose(coords, source, rtol=0, atol=0.001) for source in expected[frame]):
            raise ValueError(f'TrackLab frame {frame} includes a detection outside the shared cache')
        if pd.notna(row.track_id):
            x, y, width, height = row.bbox_ltwh
            rows.append((frame, int(row.track_id), x, y, width, height, row.bbox_conf, -1, -1, -1))
    if not rows or len({row[:2] for row in rows}) != len(rows):
        raise ValueError('Missing tracks or repeated ID in a frame')
    target = run / 'trackers' / 'CAMELTrack' / 'data' / f'{sequence}.txt'
    target.parent.mkdir(parents=True, exist_ok=False)
    with target.open('w', newline='') as file:
        csv.writer(file, lineterminator='\n').writerows(sorted(rows))
    print(f'1/25 sequences, {length} frames, {len(rows)} exported tracks', flush=True)

    evaluator = trackeval.Evaluator({'USE_PARALLEL': False, 'PLOT_CURVES': False})
    dataset = trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER': str(SPLIT), 'TRACKERS_FOLDER': str(run / 'trackers'),
        'OUTPUT_FOLDER': str(run / 'trackeval'), 'TRACKERS_TO_EVAL': ['CAMELTrack'],
        'BENCHMARK': 'DanceTrack', 'SPLIT_TO_EVAL': 'val', 'SKIP_SPLIT_FOL': True,
        'SEQ_INFO': {sequence: length}, 'DO_PREPROC': True,
    })
    evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])


if __name__ == '__main__':
    main()
