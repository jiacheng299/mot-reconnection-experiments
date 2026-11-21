"""Generate a shared YOLOX-X person detection cache for MOT17-10."""

import argparse
from configparser import ConfigParser
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'p2'
SEQUENCE = 'MOT17-10-FRCNN'
YOLOX_COMMIT = '6ddff4824372906469a7fae2dc3206c7aa4bbaee'
WEIGHTS_URL = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth'
SIZE = (640, 640)
MIN_SCORE = 0.01
NMS_IOU = 0.65


def load_image(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    return image


def sequence_info(sequence=SEQUENCE):
    config = ConfigParser()
    with (DATA / 'gt' / sequence / 'seqinfo.ini').open() as file:
        config.read_file(file)
    return config['Sequence']


def file_sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load_yolox():
    import torch

    repo = ROOT / 'vendor' / 'YOLOX'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != YOLOX_COMMIT:
        raise ValueError(f'Unexpected YOLOX commit {commit}')
    sys.path.insert(0, str(repo))
    from yolox.data.data_augment import preproc
    from yolox.exp import get_exp

    model = get_exp(exp_name='yolox-x').get_model()
    weights = DATA / 'yolox_x.pth'
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True)['model'])
    return model.cuda().eval(), preproc


def person_boxes(prediction, ratio, width, height, nms_iou=NMS_IOU):
    import torch
    from torchvision.ops import nms

    # COCO index 0 is person. Keep person scores even when another class ranks higher.
    scores = prediction[:, 4] * prediction[:, 5]
    mask = scores >= MIN_SCORE
    boxes, scores = prediction[mask, :4], scores[mask]
    half_size = boxes[:, 2:] / 2
    xyxy = torch.cat((boxes[:, :2] - half_size, boxes[:, :2] + half_size), dim=1)
    selected = nms(xyxy, scores, nms_iou)
    xyxy = xyxy[selected] / ratio
    xyxy[:, 0::2].clamp_(0, width)
    xyxy[:, 1::2].clamp_(0, height)
    return torch.cat((xyxy, scores[selected, None]), dim=1).cpu().numpy()


def write_boxes(writer, frame, boxes):
    count = 0
    for x1, y1, x2, y2, score in boxes:
        if x2 > x1 and y2 > y1:
            writer.writerow((frame, -1, f'{x1 + 1:.6f}', f'{y1 + 1:.6f}',
                             f'{x2 - x1:.6f}', f'{y2 - y1:.6f}', f'{score:.8f}'))
            count += 1
    return count


def detect(run, sequence=SEQUENCE):
    import torch
    import torchvision

    info = sequence_info(sequence)
    directory = DATA / 'gt' / sequence / info['imDir']
    width, height = info.getint('imWidth'), info.getint('imHeight')
    length = info.getint('seqLength')
    device = torch.device('cuda')
    weights = DATA / 'yolox_x.pth'
    model, preproc = load_yolox()

    destination = run / 'detections' / f'{sequence}.txt'
    destination.parent.mkdir(parents=True, exist_ok=False)
    staged = destination.with_suffix('.tmp')
    count = 0
    source_digest = hashlib.sha256()
    with staged.open('w', newline='') as file, torch.inference_mode():
        writer = csv.writer(file, lineterminator='\n')
        for frame in range(1, length + 1):
            path = directory / f'{frame:06d}{info["imExt"]}'
            image = load_image(path)
            source_digest.update(bytes.fromhex(file_sha256(path)))
            if image.shape[:2] != (height, width):
                raise ValueError(f'Incorrect dimensions for frame {frame}')
            array, ratio = preproc(image, SIZE)
            prediction = model(torch.from_numpy(array).unsqueeze(0).to(device))[0]
            count += write_boxes(writer, frame, person_boxes(prediction, ratio, width, height))
            if frame % 100 == 0 or frame == length:
                print(f'{frame}/{length} frames, {count} detections', flush=True)
    staged.rename(destination)
    manifest = {
        'purpose': 'Single-sequence detection cache; official COCO checkpoint without MOT fine-tuning',
        'sequence': sequence, 'frames': length, 'frame_content_sha256': source_digest.hexdigest(),
        'detector': 'YOLOX-X', 'code_commit': YOLOX_COMMIT, 'weights_source': WEIGHTS_URL,
        'weights_sha256': file_sha256(weights), 'input_size_hw': SIZE, 'precision': 'float32',
        'person_score_floor': MIN_SCORE, 'nms_iou': NMS_IOU, 'count': count,
        'detections_sha256': file_sha256(destination), 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'gpu': torch.cuda.get_device_name(device),
    }
    destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--sequence', choices=(SEQUENCE, 'MOT17-05-FRCNN'), default=SEQUENCE)
    args = parser.parse_args()
    detect(args.run.resolve(), args.sequence)
