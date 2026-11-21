"""Build a shared cache and score the motion baselines on official DanceTrack val."""

import argparse
from configparser import ConfigParser
import csv
import hashlib
from itertools import batched
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from zlib import crc32

import numpy as np

from detect_mot17_dev import (DATA, ROOT, MIN_SCORE, NMS_IOU, SIZE, WEIGHTS_URL,
                              YOLOX_COMMIT, file_sha256, load_image, load_yolox,
                              person_boxes, write_boxes)


DANCE = DATA / 'dance'
SPLIT = DANCE / 'val'
BATCH = 8


def prepare():
    with ZipFile(DANCE / 'val.zip') as archive:
        members = []
        for entry in archive.infolist():
            path = Path(entry.filename)
            if not entry.is_dir() and len(path.parts) >= 3 and path.parts[0] == 'val' and path.parts[1].startswith('dancetrack'):
                if '..' in path.parts or path.is_absolute():
                    raise ValueError(f'Invalid member {path}')
                members.append(entry)
        for entry in members:
            target = DANCE / entry.filename
            if target.is_file() and crc32(target.read_bytes()) == entry.CRC:
                continue
            archive.extract(entry, DANCE)
    print(f'Prepared {len(members)} files with ZIP CRC checks', flush=True)


def sequences():
    paths = sorted(SPLIT.glob('dancetrack*/seqinfo.ini'))
    if len(paths) != 25:
        raise ValueError(f'Expected all 25 official val sequences, found {len(paths)}')
    for path in paths:
        cfg = ConfigParser()
        with path.open() as file:
            cfg.read_file(file)
        info = cfg['Sequence']
        if info['name'] != path.parent.name:
            raise ValueError(f'Sequence name does not match {path.parent.name}')
        yield path.parent, info


def detect(run, available_only=False):
    import torch

    destination = run / 'detections'
    destination.mkdir(parents=True, exist_ok=True)
    model, preproc = load_yolox()
    model.half()
    weights_sha = file_sha256(DATA / 'yolox_x.pth')
    with torch.inference_mode():
        for folder, info in sequences():
            path = destination / f'{folder.name}.txt'
            meta = path.with_suffix('.json')
            if path.is_file() and meta.is_file():
                record = json.loads(meta.read_text(encoding='utf-8'))
                if record['detections_sha256'] != file_sha256(path) or record['weights_sha256'] != weights_sha:
                    raise ValueError(f'Existing cache metadata does not match {path}')
                continue
            length = info.getint('seqLength')
            images = sorted((folder / info['imDir']).glob(f'*{info["imExt"]}'), key=lambda p: int(p.stem))
            if [int(p.stem) for p in images] != list(range(1, length + 1)):
                if available_only:
                    continue
                raise ValueError(f'Incomplete or unaligned frames in {folder.name}')
            width, height = info.getint('imWidth'), info.getint('imHeight')
            digest = hashlib.sha256()
            count = 0
            staged = path.with_suffix('.tmp')
            with staged.open('w', newline='') as file:
                writer = csv.writer(file, lineterminator='\n')
                for batch in batched(images, BATCH):
                    image_arrays, ratios = [], []
                    for image in batch:
                        digest.update(bytes.fromhex(file_sha256(image)))
                        pixels = load_image(image)
                        if pixels.shape[:2] != (height, width):
                            raise ValueError(f'Unexpected image dimensions for {image}')
                        array, ratio = preproc(pixels, SIZE)
                        image_arrays.append(array)
                        ratios.append(ratio)
                    predictions = model(torch.from_numpy(np.stack(image_arrays)).to('cuda', torch.float16)).float()
                    for image, pred, ratio in zip(batch, predictions, ratios, strict=True):
                        count += write_boxes(writer, int(image.stem), person_boxes(pred, ratio, width, height))
            staged.replace(path)
            record = {'sequence': folder.name, 'frames': length, 'count': count,
                      'detector': 'YOLOX-X COCO', 'source': WEIGHTS_URL, 'commit': YOLOX_COMMIT,
                      'weights_sha256': weights_sha, 'frame_content_sha256': digest.hexdigest(),
                      'detections_sha256': file_sha256(path), 'min_score': MIN_SCORE,
                      'nms_iou': NMS_IOU, 'input_size_hw': SIZE, 'precision': 'float16',
                      'batch_size': BATCH}
            meta.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            print(f'{folder.name}: {length} frames, {count} candidates', flush=True)


def score(run, available_only=False):
    import trackeval
    from run_p1 import load_tracker, predict, read_detections
    from score_ocsort_dev import COMMIT, OPTIONS
    import subprocess

    repo = ROOT / 'vendor' / 'OC_SORT'
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('Unexpected OC-SORT commit')
    sys.path.insert(0, str(repo))
    from trackers.ocsort_tracker.ocsort import OCSort

    seq_info = {}
    byte_track = load_tracker(run)
    from yolox.tracker.basetrack import BaseTrack

    for folder, info in sequences():
        name, length = folder.name, info.getint('seqLength')
        cache = run / 'detections' / f'{name}.txt'
        if available_only and not cache.with_suffix('.json').is_file():
            continue
        seq_info[name] = length
        record = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
        if record['frames'] != length or file_sha256(cache) != record['detections_sha256']:
            raise ValueError(f'Invalid cache for {name}')
        print(f'Using {name} shared SHA-256 {record["detections_sha256"]}', flush=True)
        BaseTrack._count = 0
        predict(run, info, cache, tracker_class=byte_track)
        tracker = OCSort(**OPTIONS)
        detections = read_detections(cache, length)
        size = (info.getint('imHeight'), info.getint('imWidth'))
        path = run / 'trackers' / 'OC-SORT' / 'data' / f'{name}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as file:
            writer = csv.writer(file, lineterminator='\n')
            for frame in range(1, length + 1):
                boxes = np.asarray(detections[frame], dtype=float).reshape(-1, 5)
                for x, y, right, bottom, identity in tracker.update(boxes, size, size):
                    width, height = right - x, bottom - y
                    if height > 0 and width * height > 10 and width / height <= 1.6:
                        writer.writerow((frame, int(identity), f'{x:.4f}', f'{y:.4f}',
                                         f'{width:.4f}', f'{height:.4f}', 1, -1, -1, -1))

    if not seq_info:
        raise ValueError('No verified DanceTrack detection caches are available')
    print(f'Evaluation coverage: {len(seq_info)}/25 official val sequences', flush=True)
    evaluator = trackeval.Evaluator({'USE_PARALLEL': False, 'PLOT_CURVES': False})
    dataset = trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER': str(SPLIT), 'TRACKERS_FOLDER': str(run / 'trackers'),
        'OUTPUT_FOLDER': str(run / 'trackeval'), 'TRACKERS_TO_EVAL': ['ByteTrack', 'OC-SORT'],
        'BENCHMARK': 'DanceTrack', 'SPLIT_TO_EVAL': 'val', 'SKIP_SPLIT_FOL': True,
        'SEQ_INFO': seq_info, 'DO_PREPROC': True,
    })
    evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'detect', 'score'))
    parser.add_argument('run', type=Path, nargs='?', default=ROOT / 'runs' / 'p2-dance-001')
    parser.add_argument('--available-only', action='store_true', help='operate on complete inputs only and report coverage')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'detect':
        detect(args.run.resolve(), args.available_only)
    else:
        score(args.run.resolve(), args.available_only)
