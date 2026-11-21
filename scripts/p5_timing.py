"""Measure association-only and image-to-track latency with batch size one."""

import argparse
from functools import partial
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.detect_mot17_dev import load_image, person_boxes
from scripts.p5_dance_data import NMS_IOU, frame_paths, load_detector, normalize, sequences
from scripts.p5_dance_eval import GATES, cached_detections, hand_score, set_gate
from scripts.p5_train import load_ranker
from scripts.score_p3_c1 import track_sequence


CLIP_LENGTH = 120
WARMUP = 20
REPEATS = 3


class LiveDetections:
    def __init__(self, images, model, preproc, width, height):
        self.images = images
        self.model = model
        self.preproc = preproc
        self.width = width
        self.height = height

    @torch.inference_mode()
    def get(self, frame, _default):
        array, ratio = normalize(load_image(self.images[frame - 1]), self.preproc)
        prediction = self.model(torch.from_numpy(array[None]).to('cuda', torch.float16))[0].float()
        boxes = person_boxes(prediction, ratio, self.width, self.height, NMS_IOU)
        return [(float(x1) + 1, float(y1) + 1, float(x2) + 1, float(y2) + 1, float(score))
                for x1, y1, x2, y2, score in boxes if x1 < x2 and y1 < y2]


def summaries(frames):
    array = np.asarray(frames)
    times = array[:, 0] / 1e6
    return {'mean_ms': float(times.mean()), 'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)), 'frames': len(times),
            'candidates_per_group': float(array[:, 1].sum() / max(1, array[:, 2].sum())),
            'effective_fps': float(1000 / times.mean())}


def benchmark(cache, checkpoint, gate, hand_weight, budgets):
    set_gate(gate)
    torch.set_num_threads(1)
    joint, _ = load_ranker(checkpoint)
    model, preproc = load_detector()
    val = list(sequences('val'))
    clips = []
    for folder, info in (val[0], val[len(val) // 2], val[-1]):
        paths = frame_paths(folder, info)[:CLIP_LENGTH]
        if len(paths) != CLIP_LENGTH:
            raise ValueError(f'Unexpected short timing clip: {folder}')
        _, cached = cached_detections(cache / 'val' / f'{folder.name}.txt', 'val', folder.name, info.getint('seqLength'))
        live = LiveDetections(paths, model, preproc, info.getint('imWidth'), info.getint('imHeight'))
        clips.append((folder.name, cached, live))
    rankers = {'geometry': None, 'manual-pair': partial(hand_score, weight=hand_weight), 'learned': joint}
    results = []
    with TemporaryDirectory() as scratch:
        path = Path(scratch)
        for budget in budgets:
            for label, ranker in rankers.items():
                for stream in ('association', 'end_to_end'):
                    for repeat in range(REPEATS):
                        frames = []
                        for sequence, cached, live in clips:
                            source = live if stream == 'end_to_end' else cached
                            sample = []
                            run = path / f'{budget}-{label}-{stream}-{repeat}-{sequence}'
                            track_sequence(run, source, CLIP_LENGTH, sequence, label, ranker,
                                           candidate_limit=budget, timings=sample)
                            frames.extend(sample[WARMUP:])
                        record = {'budget': budget, 'scorer': label, 'scope': stream, 'repeat': repeat,
                                  **summaries(frames)}
                        results.append(record)
                        print(json.dumps(record), flush=True)
    return {'clips': [sample[0] for sample in clips], 'frames_per_clip': CLIP_LENGTH,
            'excluded_warmup_per_clip': WARMUP, 'gpu': torch.cuda.get_device_name(),
            'detector_precision': 'float16', 'association_precision': 'float32 model / float64 geometry',
            'includes_image_decode': True, 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--gate', required=True, choices=GATES)
    parser.add_argument('--hand-weight', type=float, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=(1, 4, 8, 16, 32))
    args = parser.parse_args()
    value = benchmark(args.cache, args.checkpoint, args.gate, args.hand_weight, args.budgets)
    if args.output.exists():
        raise FileExistsError(args.output)
    staged = args.output.with_suffix('.tmp')
    with staged.open('w', encoding='utf-8') as file:
        json.dump(value, file, indent=2)
    staged.replace(args.output)
