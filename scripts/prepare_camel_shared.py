"""Adapt the shared DanceTrack cache to TrackLab's 10-column input format."""

import argparse
import csv
import json

from detect_mot17_dev import ROOT, file_sha256
from run_dance_dev import sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence')
    args = parser.parse_args()
    for folder, info in sequences():
        if args.sequence and args.sequence != folder.name:
            continue
        cache = ROOT / 'runs' / 'p2-dance-001' / 'detections' / f'{folder.name}.txt'
        record = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
        digest = file_sha256(cache)
        if digest != record['detections_sha256'] or record['frames'] != info.getint('seqLength'):
            raise ValueError(f'Invalid shared cache for {folder.name}')
        with cache.open(newline='') as file:
            rows = [row for row in csv.reader(file) if float(row[6]) >= 0.4]
        if not rows or rows[0][0] != '1':
            raise ValueError(f'TrackLab expects the first detection at frame 1 for {folder.name}')
        path = folder / 'det' / 'cghtrack_p2_yolox_conf040.txt'
        path.parent.mkdir(exist_ok=True)
        with path.open('w', newline='') as file:
            csv.writer(file, lineterminator='\n').writerows([*row, -1, -1, -1] for row in rows)
        print(folder.name, digest, len(rows), file_sha256(path), flush=True)


if __name__ == '__main__':
    main()
