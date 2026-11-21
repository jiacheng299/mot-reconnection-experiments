"""Recreate the shared MOT17-10 cache and derive TrackTrack's two NMS views."""

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path

import torch

from detect_mot17_dev import (DATA, SEQUENCE, SIZE, file_sha256, load_image, load_yolox,
                              person_boxes, sequence_info, write_boxes)
from score_ocsort_dev import CACHE_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    model, preproc = load_yolox()
    info = sequence_info()
    length = info.getint('seqLength')
    width, height = info.getint('imWidth'), info.getint('imHeight')
    directory = DATA / 'gt' / SEQUENCE / info['imDir']
    thresholds = (0.65, 0.80, 0.95)
    paths = {iou: run / f'nms-{iou:.2f}.txt' for iou in thresholds}
    counts = dict.fromkeys(thresholds, 0)
    with ExitStack() as stack, torch.inference_mode():
        writers = {iou: csv.writer(stack.enter_context(path.open('w', newline='')), lineterminator='\n')
                   for iou, path in paths.items()}
        for frame in range(1, length + 1):
            image = load_image(directory / f'{frame:06d}{info["imExt"]}')
            array, ratio = preproc(image, SIZE)
            prediction = model(torch.from_numpy(array).unsqueeze(0).cuda())[0]
            for iou in thresholds:
                boxes = person_boxes(prediction, ratio, width, height, nms_iou=iou)
                counts[iou] += write_boxes(writers[iou], frame, boxes)
            if frame % 100 == 0 or frame == length:
                print(f'{frame}/{length} frames, {counts}', flush=True)
    digests = {iou: file_sha256(path) for iou, path in paths.items()}
    if digests[0.65] != CACHE_SHA256:
        raise ValueError('Reference NMS 0.65 does not reproduce the previous cache')
    (run / 'inputs.json').write_text(json.dumps({
        'sequence': SEQUENCE, 'frames': length, 'weights_sha256': file_sha256(DATA / 'yolox_x.pth'),
        'reference_cache_verified': True,
        'views': {f'{iou:.2f}': {'count': counts[iou], 'sha256': digests[iou]} for iou in thresholds},
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
