"""Verify official DanceTrack inputs and generate frozen D-Dance detections."""

import argparse
from configparser import ConfigParser
import csv
import hashlib
from itertools import batched
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from zipfile import ZipFile
from zlib import crc32

import numpy as np

from scripts.detect_mot17_dev import DATA, ROOT, YOLOX_COMMIT, file_sha256, load_image, person_boxes, write_boxes


DANCE = DATA / 'dance'
WEIGHT_FILE = DANCE / 'bytetrack_model.pth.tar'
WEIGHT_SOURCE = ('https://huggingface.co/noahcao/dancetrack_models/resolve/'
                 '6487fc12a2fe13db44506592324cea83270f5812/bytetrack_models/bytetrack_model.pth.tar')
WEIGHT_HASH = 'b8d1afba08f801f3fe2cb122faf3fc9af6c7856405a0da94cf91cdd5eb9b3321'
TRAIN_ARCHIVES = {
    'train1.zip': '70a66f10d8d71df94d03059fc4f966cb719e61e8e7136d2886fd6345ed1ff6dd',
    'train2.zip': 'f676e8c6ac2a2d7566f1702d588311eafd4c90cc5416bb70f36a192d1e365af4',
}
SIZE = (800, 1440)
BATCH = 4
NMS_IOU = 0.70
MIN_SCORE = 0.01
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
NAME = re.compile(r'dancetrack\d{4}')


def train_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        return None
    parts = path.parts
    for index, item in enumerate(parts):
        if not NAME.fullmatch(item):
            continue
        if not parts[:index] or parts[index - 1] not in ('train', 'train1', 'train2'):
            return None
        tail = parts[index + 1:]
        if tail in (('seqinfo.ini',), ('gt', 'gt.txt')) or (
            len(tail) == 2 and tail[0] == 'img1' and re.fullmatch(r'\d{8}\.jpg', tail[1])
        ):
            return parts[index:]
        return None
    return None


def official_names(split):
    lines = (DANCE / f'official-{split}-seqmap.txt').read_text(encoding='utf-8-sig').split()
    names = lines[1:]
    expected = {'train': 40, 'val': 25}[split]
    if lines[0] != 'name' or len(set(names)) != len(names) or len(names) != expected or not all(
        NAME.fullmatch(name) for name in names
    ):
        raise ValueError(f'Invalid official {split} seqmap')
    return sorted(names)


def sequences(split):
    names = official_names(split)
    root = DANCE / split
    found = {path.parent.name for path in root.glob('dancetrack*/seqinfo.ini')}
    if found != set(names):
        raise ValueError(f'{split} does not match the official list: missing={sorted(set(names) - found)}, extra={sorted(found - set(names))}')
    for name in names:
        folder = root / name
        config = ConfigParser()
        with (folder / 'seqinfo.ini').open() as file:
            config.read_file(file)
        info = config['Sequence']
        if info['name'] != name or info.getint('seqLength') < 1 or not (folder / 'gt' / 'gt.txt').is_file():
            raise ValueError(f'Incomplete sequence metadata in {folder}')
        yield folder, info


def frame_paths(folder, info):
    paths = sorted((folder / info['imDir']).glob(f'*{info["imExt"]}'))
    if [int(path.stem) for path in paths] != list(range(1, info.getint('seqLength') + 1)):
        raise ValueError(f'Missing or unexpected frames in {folder}')
    return paths


def prepare_train():
    names = set(official_names('train'))
    seen = set()
    for archive_name, digest in TRAIN_ARCHIVES.items():
        source = DANCE / archive_name
        if file_sha256(source) != digest:
            raise ValueError(f'Incorrect archive digest: {source}')
        with ZipFile(source) as archive:
            for entry in archive.infolist():
                if entry.is_dir() or (parts := train_member(entry.filename)) is None:
                    continue
                if parts[0] not in names or parts in seen:
                    raise ValueError(f'Unexpected or repeated training archive entry: {entry.filename}')
                seen.add(parts)
                target = DANCE.joinpath('train', *parts)
                if target.is_file() and target.stat().st_size == entry.file_size and crc32(target.read_bytes()) == entry.CRC:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                staged = target.with_suffix('.tmp')
                with archive.open(entry) as data, staged.open('wb') as file:
                    shutil.copyfileobj(data, file)
                staged.replace(target)
        print(f'Verified and unpacked {archive_name}', flush=True)
    total = sum(len(frame_paths(folder, info)) for folder, info in sequences('train'))
    print(f'Prepared {len(names)} training sequences with {total} frames', flush=True)


def load_detector():
    import torch

    if file_sha256(WEIGHT_FILE) != WEIGHT_HASH:
        raise ValueError('D-Dance weight digest does not match the publisher')
    repo = ROOT / 'vendor' / 'YOLOX'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != YOLOX_COMMIT:
        raise ValueError(f'Unexpected YOLOX commit {commit}')
    sys.path.insert(0, str(repo))
    if 'yolox' in sys.modules and str(repo / 'yolox') not in sys.modules['yolox'].__path__:
        sys.modules['yolox'].__path__.append(str(repo / 'yolox'))
    from yolox.data.data_augment import preproc
    from yolox.exp import get_exp
    from yolox.utils import fuse_model

    spec = get_exp(exp_name='yolox-x')
    spec.num_classes = 1
    model = spec.get_model()
    checkpoint = torch.load(WEIGHT_FILE, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model = fuse_model(model.eval()).cuda().half()
    return model, preproc


def normalize(pixels, preproc):
    array, ratio = preproc(pixels, SIZE)
    return (array[::-1] / 255.0 - MEAN) / STD, ratio


def cache_config():
    return {'detector': 'D-Dance YOLOX-X', 'weights_source': WEIGHT_SOURCE, 'weights_sha256': WEIGHT_HASH,
            'code_commit': YOLOX_COMMIT, 'input_size_hw': list(SIZE), 'precision': 'float16',
            'preprocessing': 'RGB ImageNet mean/std', 'min_score': MIN_SCORE, 'nms_iou': NMS_IOU,
            'batch_size': BATCH, 'fused': True}


def verify_cache(path, split, sequence, length):
    record = json.loads(path.with_suffix('.json').read_bytes())
    expected = {'split': split, 'sequence': sequence, 'frames': length, **cache_config()}
    if any(record.get(key) != value for key, value in expected.items()) or (
        record['detections_sha256'] != file_sha256(path)
    ):
        raise ValueError(f'Cache does not match frozen detector or sequence: {path}')
    return record


def detect(output, split, only=None):
    import torch
    import torchvision
    import cv2

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    destination = output / split
    destination.mkdir(parents=True, exist_ok=True)
    complete = list(sequences(split))
    if only is not None and only not in {folder.name for folder, _ in complete}:
        raise ValueError(f'Sequence {only} is absent from official {split}')
    model, preproc = load_detector()
    with torch.inference_mode():
        for folder, info in complete:
            if only is not None and folder.name != only:
                continue
            path = destination / f'{folder.name}.txt'
            meta = path.with_suffix('.json')
            length = info.getint('seqLength')
            if path.is_file() and meta.is_file():
                verify_cache(path, split, folder.name, length)
                continue
            width, height = info.getint('imWidth'), info.getint('imHeight')
            digest = hashlib.sha256()
            count = 0
            staged = path.with_suffix('.tmp')
            with staged.open('w', newline='') as file:
                writer = csv.writer(file, lineterminator='\n')
                for batch in batched(frame_paths(folder, info), BATCH):
                    frames, ratios = [], []
                    for image in batch:
                        digest.update(bytes.fromhex(file_sha256(image)))
                        pixels = load_image(image)
                        if pixels.shape[:2] != (height, width):
                            raise ValueError(f'Unexpected image dimensions: {image}')
                        array, ratio = normalize(pixels, preproc)
                        frames.append(array)
                        ratios.append(ratio)
                    prediction = model(torch.from_numpy(np.stack(frames)).to('cuda', torch.float16)).float()
                    for image, pred, ratio in zip(batch, prediction, ratios, strict=True):
                        count += write_boxes(writer, int(image.stem), person_boxes(pred, ratio, width, height, NMS_IOU))
            staged.replace(path)
            record = {'sequence': folder.name, 'split': split, 'frames': length, 'count': count,
                      'frame_content_sha256': digest.hexdigest(), 'detections_sha256': file_sha256(path),
                      **cache_config(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
                      'gpu': torch.cuda.get_device_name()}
            meta.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            print(f'{split}/{folder.name}: {length} frames, {count} detections', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare-train')
    cache = commands.add_parser('detect')
    cache.add_argument('output', type=Path)
    cache.add_argument('--split', choices=('train', 'val'), required=True)
    cache.add_argument('--sequence')
    args = parser.parse_args()
    if args.command == 'prepare-train':
        prepare_train()
    else:
        detect(args.output, args.split, args.sequence)
