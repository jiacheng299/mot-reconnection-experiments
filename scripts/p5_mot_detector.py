"""Fit and cache D-MOT using only the project's allowed MOT17 training videos."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.detect_mot17_dev import (
    DATA, ROOT, MIN_SCORE, NMS_IOU, SIZE, WEIGHTS_URL, YOLOX_COMMIT, file_sha256, load_image,
    person_boxes, sequence_info, write_boxes,
)
from scripts.p4_supervision import load_pedestrian_truth


TRAIN = tuple(f'MOT17-{i:02d}-FRCNN' for i in (2, 4, 5, 9))
VAL = ('MOT17-10-FRCNN',)
BATCH = 4
STRIDE = 5
EPOCHS = 2
SEED = 0
LEARNING_RATE = 0.0001


def spec_and_preproc():
    repo = ROOT / 'vendor' / 'YOLOX'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != YOLOX_COMMIT:
        raise ValueError(f'Unexpected YOLOX code revision {commit}')
    sys.path.insert(0, str(repo))
    if 'yolox' in sys.modules and str(repo / 'yolox') not in sys.modules['yolox'].__path__:
        sys.modules['yolox'].__path__.append(str(repo / 'yolox'))
    from yolox.exp import get_exp
    from yolox.data.data_augment import preproc

    spec = get_exp(exp_name='yolox-x')
    spec.num_classes = 1
    return spec, preproc


def prepare_labels(boxes, ratio, width, height):
    labels = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = np.clip([x1 - 1, x2 - 1], 0, width)
        y1, y2 = np.clip([y1 - 1, y2 - 1], 0, height)
        if x2 > x1 and y2 > y1:
            labels.append((0, (x1 + x2) / 2 * ratio, (y1 + y2) / 2 * ratio,
                           (x2 - x1) * ratio, (y2 - y1) * ratio))
    return labels


def dataset():
    samples, manifest = [], {}
    for name in TRAIN:
        info = sequence_info(name)
        length = info.getint('seqLength')
        root = DATA / 'gt' / name
        source = root / 'gt' / 'gt.txt'
        truth = load_pedestrian_truth(source)
        selected = list(range(1, length + 1, STRIDE))
        paths = [root / info['imDir'] / f'{frame:06d}{info["imExt"]}' for frame in selected]
        if not all(path.is_file() for path in paths):
            raise ValueError(f'Incomplete training images for {name}')
        image_hash = hashlib.sha256()
        for frame, path in zip(selected, paths, strict=True):
            image_hash.update(bytes.fromhex(file_sha256(path)))
            samples.append((path, list(truth[frame].values())))
        manifest[name] = {'frames': length, 'sampled_frames': selected, 'gt_sha256': file_sha256(source),
                          'sampled_image_sha256': image_hash.hexdigest()}
    return samples, manifest


def train(output):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    samples, training = dataset()
    spec, preproc = spec_and_preproc()
    model = spec.get_model()
    source = DATA / 'yolox_x.pth'
    original = torch.load(source, map_location='cpu', weights_only=True)['model']
    expected = model.state_dict()
    loaded = {key: tensor if tensor.shape == expected[key].shape else tensor[:1]
              for key, tensor in original.items() if not key.endswith('num_batches_tracked')}
    missing, extra = model.load_state_dict(loaded, strict=False)
    if extra or not all(key.endswith('num_batches_tracked') for key in missing):
        raise ValueError(f'Unexpected differences loading COCO weights: {missing}, {extra}')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('head.'))
    model.cuda().train()
    model.backbone.eval()
    optim = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(samples))
        losses = []
        for start in range(0, len(samples), BATCH):
            images, targets = [], []
            for index in order[start:start+BATCH]:
                path, boxes = samples[index]
                pixels = load_image(path)
                array, ratio = preproc(pixels, SIZE)
                images.append(array)
                targets.append(prepare_labels(boxes, ratio, pixels.shape[1], pixels.shape[0]))
            target = np.zeros((len(targets), max(1, max(map(len, targets))), 5), dtype=np.float32)
            for index, labels in enumerate(targets):
                if labels:
                    target[index, :len(labels)] = labels
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(torch.from_numpy(np.stack(images)).cuda(), torch.from_numpy(target).cuda())['total_loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite D-MOT training loss at epoch {epoch + 1}, sample {start}')
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.detach()))
            if len(losses) % 50 == 0:
                print(f'D-MOT epoch {epoch + 1}/{EPOCHS}, {start + len(images)}/{len(samples)} images', flush=True)
        history.append({'epoch': epoch + 1, 'mean_loss': float(np.mean(losses)), 'steps': len(losses)})
        print(json.dumps(history[-1]), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as file:
        torch.save({'model': {name: value.cpu() for name, value in model.state_dict().items()}}, file)
    output.with_suffix('.json').write_text(json.dumps({
        'detector': 'D-MOT YOLOX-X', 'train': training, 'history': history, 'trainable': 'detection head',
        'seed': SEED, 'input_size_hw': SIZE, 'batch_size': BATCH, 'stride': STRIDE,
        'learning_rate': LEARNING_RATE, 'initial_weights_source': WEIGHTS_URL,
        'initial_weights_sha256': file_sha256(source), 'weights_sha256': file_sha256(output),
        'code_commit': YOLOX_COMMIT,
    }, indent=2) + '\n', encoding='utf-8')


def cache_weights(weight):
    record = json.loads(weight.with_suffix('.json').read_bytes())
    if set(record['train']) != set(TRAIN) or record['weights_sha256'] != file_sha256(weight):
        raise ValueError('D-MOT has an incorrect weight or training manifest')
    return record


def verify_cache(path, split, name, length):
    record = json.loads(path.with_suffix('.json').read_bytes())
    if (name not in (TRAIN if split == 'train' else VAL) or record['sequence'] != name or
            record['split'] != split or record['frames'] != length or record['detector'] != 'D-MOT YOLOX-X' or
            record['detections_sha256'] != file_sha256(path)):
        raise ValueError(f'Invalid D-MOT cache {path}')
    return record


def detect(weight, destination, split):
    import torch
    from itertools import batched

    torch.set_num_threads(1)
    spec, preproc = spec_and_preproc()
    weights = cache_weights(weight)
    model = spec.get_model()
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True)['model'])
    model.cuda().eval().half()
    folder = destination / split
    folder.mkdir(parents=True, exist_ok=True)
    for name in TRAIN if split == 'train' else VAL:
        info = sequence_info(name)
        length, width, height = (info.getint(key) for key in ('seqLength', 'imWidth', 'imHeight'))
        target = folder / f'{name}.txt'
        if target.is_file() and target.with_suffix('.json').is_file():
            if verify_cache(target, split, name, length)['weights_sha256'] != weights['weights_sha256']:
                raise ValueError('Existing D-MOT cache has another detector')
            continue
        staged = target.with_suffix('.tmp')
        digest = hashlib.sha256()
        source = DATA / 'gt' / name / info['imDir']
        count = 0
        with staged.open('w', newline='') as file, torch.inference_mode():
            writer = csv.writer(file, lineterminator='\n')
            for batch in batched(range(1, length + 1), BATCH):
                images, ratios = [], []
                for frame in batch:
                    image = source / f'{frame:06d}{info["imExt"]}'
                    digest.update(bytes.fromhex(file_sha256(image)))
                    pixels = load_image(image)
                    if pixels.shape[:2] != (height, width):
                        raise ValueError(f'Frame dimensions changed: {image}')
                    array, ratio = preproc(pixels, SIZE)
                    images.append(array)
                    ratios.append(ratio)
                predictions = model(torch.from_numpy(np.stack(images)).to('cuda', torch.float16)).float()
                for frame, prediction, ratio in zip(batch, predictions, ratios, strict=True):
                    count += write_boxes(writer, frame, person_boxes(prediction, ratio, width, height))
        staged.replace(target)
        target.with_suffix('.json').write_text(json.dumps({
            'detector': weights['detector'], 'weights_sha256': weights['weights_sha256'], 'split': split,
            'sequence': name, 'frames': length, 'count': count, 'detections_sha256': file_sha256(target),
            'frame_content_sha256': digest.hexdigest(), 'min_score': MIN_SCORE, 'nms_iou': NMS_IOU,
            'input_size_hw': SIZE, 'batch_size': BATCH, 'precision': 'float16',
        }, indent=2) + '\n', encoding='utf-8')
        print(f'{split}/{name}: {length} frames, {count} detections', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('output', type=Path)
    cache = commands.add_parser('detect')
    cache.add_argument('checkpoint', type=Path)
    cache.add_argument('output', type=Path)
    cache.add_argument('--split', choices=('train', 'val'), required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        train(args.output)
    else:
        detect(args.checkpoint, args.output, args.split)
