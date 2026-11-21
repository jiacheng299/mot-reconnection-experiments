"""Evaluate the author's OC-SORT on the frozen MOT17-10 YOLOX-X cache."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from detect_mot17_dev import DATA, ROOT, SEQUENCE, file_sha256, sequence_info
from run_p1 import evaluate, read_detections
from score_mot17_dev import preview


COMMIT = '8462e7e729a93ccd3bd995c0a79a890336cb3a0b'
CACHE_SHA256 = '00f0107933cac67d8ff0bc4667384cee6d1e478abb2d1e7dda409c7b98a4bda2'
OPTIONS = {'det_thresh': 0.6, 'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3,
           'delta_t': 3, 'asso_func': 'iou', 'inertia': 0.2, 'use_byte': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='new output directory')
    run = parser.parse_args().run.resolve()
    repo = ROOT / 'vendor' / 'OC_SORT'
    git = ['git', '-C', str(repo)]
    commit = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output([*git, 'status', '--porcelain', '--', 'trackers'], text=True)
    if commit != COMMIT or changes:
        raise ValueError('Expected the pinned OC-SORT tracker source')
    sys.path.insert(0, str(repo))
    from trackers.ocsort_tracker.ocsort import OCSort

    info = sequence_info()
    length = info.getint('seqLength')
    size = (info.getint('imHeight'), info.getint('imWidth'))
    cache = ROOT / 'runs' / 'p2-002' / 'detections' / f'{SEQUENCE}.txt'
    cache_metadata = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
    if (file_sha256(cache) != CACHE_SHA256 or cache_metadata['detections_sha256'] != CACHE_SHA256
            or cache_metadata['sequence'] != SEQUENCE or cache_metadata['frames'] != length):
        raise ValueError('Input must match the detection cache used for ByteTrack')
    detections = read_detections(cache, length)
    count = sum(map(len, detections.values()))
    if count != cache_metadata['count']:
        raise ValueError('Incorrect number of cached detections')
    print(f'Using shared cache SHA-256 {CACHE_SHA256}', flush=True)

    run.mkdir(parents=True, exist_ok=False)
    (run / 'run-config.json').write_text(json.dumps({
        'sequence': SEQUENCE, 'frames': length, 'input_rows': count,
        'detections_sha256': CACHE_SHA256, 'ocsort_commit': commit, 'ocsort_options': OPTIONS,
        'device': 'cpu', 'min_box_area': 10, 'max_width_height_ratio': 1.6,
    }, indent=2) + '\n', encoding='utf-8')
    output = run / 'trackers' / 'OC-SORT' / 'data' / f'{SEQUENCE}.txt'
    output.parent.mkdir(parents=True)
    tracker = OCSort(**OPTIONS)
    tracks = defaultdict(list)
    with output.open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        for frame in range(1, length + 1):
            boxes = np.array(detections[frame], dtype=float).reshape(-1, 5)
            for x1, y1, x2, y2, identity in tracker.update(boxes, size, size):
                width, height = x2 - x1, y2 - y1
                if height <= 0 or width * height <= 10 or width / height > 1.6:
                    continue
                # The author's evaluator exports a constant MOT confidence of 1.
                writer.writerow((frame, int(identity), f'{x1:.4f}', f'{y1:.4f}',
                                 f'{width:.4f}', f'{height:.4f}', 1, -1, -1, -1))
                tracks[frame].append((int(identity), x1, y1, width, height))
    print(f'Processed {length} frames, {count} detections, '
          f'{sum(map(len, tracks.values()))} predicted boxes', flush=True)
    evaluate(run, length, SEQUENCE, DATA / 'gt', tracker='OC-SORT')
    preview(run, info, cache, tracks, tracker='OC-SORT')


if __name__ == '__main__':
    main()
