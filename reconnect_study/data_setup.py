"""Prepare traceable inputs for the public reconnection experiments."""

import argparse
import ast
from configparser import ConfigParser
import csv
import hashlib
from io import TextIOWrapper
from itertools import batched
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile


DANCE_SOURCE = 'https://github.com/DanceTrack/DanceTrack'
COCO_WEIGHT_SHA256 = '5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf'
DANCE_DETECTOR_SETTINGS = {
    'detector': 'YOLOX-X COCO', 'weights_sha256': COCO_WEIGHT_SHA256,
    'commit': '6ddff4824372906469a7fae2dc3206c7aa4bbaee', 'precision': 'float16',
    'input_size_hw': [640, 640], 'batch_size': 8, 'min_score': 0.01, 'nms_iou': 0.65,
}
PERSONPATH_SOURCE = 'https://github.com/amazon-science/tracking-dataset'
PERSONPATH_ANNOTATIONS = 'https://tracking-dataset-eccv-2022.s3.us-east-2.amazonaws.com/dataset/annotation/'
PERSONPATH_DETECTOR = ('https://aws-cv-sci-motion-public.s3.us-west-2.amazonaws.com/'
                       'PersonPath22/public_detection_fcos/mot_format.zip')
PERSONPATH_DISTRACTORS = {'fully_occluded', 'severly_occluded_person', 'person_in_background',
                          'person_in_vehicle', 'person_on_open_vehicle', 'reflection'}
BDD_LABEL_SOURCE = 'https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip'


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _frames_and_rows(path, frames):
    count = 0
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) < 7 or not 1 <= int(row[0]) <= frames:
                raise ValueError(f'Invalid MOT row at {path}: {row!r}')
            count += 1
    return count


def dance_manifest(dataset, cache_directories, *, splits, partial=False):
    """Use completed YOLOX-X COCO cache pairs in directory preference order."""
    dataset = Path(dataset).resolve()
    cache_directories = [Path(directory).resolve() for directory in cache_directories]
    records = []
    missing = []
    seen = set()
    for split in splits:
        candidates = sorted((dataset / split).glob('dancetrack*/seqinfo.ini'))
        if not candidates:
            raise ValueError(f'No sequence metadata under {dataset / split}')
        split_map = dataset / f'official-{split}-seqmap.txt'
        if split_map.is_file():
            expected = split_map.read_text(encoding='utf-8').split()[1:]
            if set(expected) != {path.parent.name for path in candidates}:
                raise ValueError(f'Sequence directories do not match {split_map}')
        for ini in candidates:
            name = ini.parent.name
            config = ConfigParser()
            with ini.open(encoding='utf-8') as handle:
                config.read_file(handle)
            info = config['Sequence']
            if info['name'] != name or name in seen:
                raise ValueError(f'Mismatched or duplicate sequence name in {ini}')
            seen.add(name)
            frames = info.getint('seqLength')
            fps = info.getfloat('frameRate')
            if frames < 1 or fps <= 0:
                raise ValueError(f'Invalid timebase in {ini}')
            gt = ini.parent / 'gt' / 'gt.txt'
            gt_rows = _frames_and_rows(gt, frames)
            detection = next((path for directory in cache_directories
                              if (path := directory / f'{name}.txt').is_file()
                              and path.with_suffix('.json').is_file()), None)
            if detection is None:
                missing.append(name)
                continue
            sidecar = detection.with_suffix('.json')
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            detection_sha256 = sha256(detection)
            if metadata['detections_sha256'] != detection_sha256:
                raise ValueError(f'Detection checksum mismatch for {detection}')
            if (metadata['sequence'] != name or metadata['frames'] != frames
                    or any(metadata.get(key) != value for key, value in DANCE_DETECTOR_SETTINGS.items())
                    or metadata['count'] != _frames_and_rows(detection, frames)):
                raise ValueError(f'Inconsistent detector metadata in {sidecar}')
            records.append({
                'name': name, 'source_id': name, 'split': split, 'frames': frames, 'fps': fps,
                'gt': str(gt), 'detections': str(detection),
                'gt_sha256': sha256(gt), 'detections_sha256': detection_sha256,
                'seqinfo_sha256': sha256(ini), 'gt_rows': gt_rows,
                'detector_metadata': str(sidecar), 'detector_run': metadata,
            })
    if missing and not partial:
        raise FileNotFoundError(f'Missing completed detection caches: {", ".join(missing)}')
    return {
        'dataset': 'DanceTrack', 'official_source': DANCE_SOURCE,
        'ground_truth': 'official per-frame person box and identity annotations',
        'detections': 'YOLOX-X inference using the released COCO weights',
        'timebase': 'Frames are one-based in both MOT files. fps is annotated frames per second.',
        'box_format': 'Pixel xywh. Detector x,y are shifted by +1 to match the MOT export. '
                      'Ground truth coordinates are copied directly from the benchmark.',
        'source_groups_verified': False,
        'source_id_scope': 'Public benchmark video identifier. Original recording or site overlaps were not resolved.',
        'expected_sequence_count': len(seen), 'missing_sequences': missing, 'sequences': records,
    }


def personpath_rows(entities, detection_lines, source_length):
    """Produce matching 1-based MOT rows on documented manual keyframes."""
    raw_frames = sorted({entity['blob']['frame_idx'] + 1 for entity in entities})
    if not raw_frames or not 1 <= raw_frames[0] <= raw_frames[-1] <= source_length:
        raise ValueError('Invalid annotation keyframes')
    keyframe_numbers = {frame: index for index, frame in enumerate(raw_frames, 1)}
    truth = []
    for entity in entities:
        frame = keyframe_numbers[entity['blob']['frame_idx'] + 1]
        labels = entity['labels']
        kind = 13 if 'crowd' in labels else 8 if PERSONPATH_DISTRACTORS.intersection(labels) else 1
        x, y, width, height = entity['bb']
        if width > 0 and height > 0:
            truth.append((frame, entity['id'], x, y, width, height, 1, kind, -1, -1))
    truth.sort(key=lambda row: (row[0], row[1]))
    detections = []
    for row in csv.reader(detection_lines):
        if not row:
            continue
        raw_frame = int(row[0])
        if len(row) < 7 or raw_frame < 1:
            raise ValueError(f'Invalid PersonPath22 detection row: {row!r}')
        if frame := keyframe_numbers.get(raw_frame):
            detections.append((frame, *row[1:]))
    return raw_frames, truth, detections


def _personpath_external_sources(download_script):
    """Read the benchmark's published filename mapping without executing its script."""
    assignments = {node.targets[0].id: node.value for node in ast.parse(
        download_script.read_text(encoding='utf-8')).body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)}
    sources = {}
    mapping = assignments['external_video_id_mapping']
    for key_node, source_node in zip(mapping.keys, mapping.values, strict=True):
        collection = ast.literal_eval(key_node)
        if collection == 'Virat':
            source_map = {original: item[0] for original, item in ast.literal_eval(
                assignments['virat_id_to_uid_and_urlid']).items()}
        else:
            source_map = ast.literal_eval(source_node)
        for original, video in source_map.items():
            if collection in {'Meva', 'Virat'}:
                # The same recording, actors and sites may recur throughout a collection.
                group = f'external:{collection}'
            elif collection == 'PathTrack':
                group = f'external:{collection}:{original.rsplit("_", 2)[0]}'
            else:
                group = f'external:{collection}:{original}'
            sources[video] = (group, collection, original)
    return sources


def _write_mot(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        csv.writer(handle, lineterminator='\n').writerows(rows)
    temporary.replace(path)


def personpath_manifest(downloads, output):
    """Join publisher annotations, splits, and public FCOS detector output."""
    splits_path = downloads / 'personpath22-splits.json'
    amodal_path = downloads / 'personpath22-anno-amodal.zip'
    detector_path = downloads / 'personpath22-detection-mot.zip'
    publisher_script = downloads / 'personpath22-publisher-download.py'
    split_map = json.loads(splits_path.read_text(encoding='utf-8'))
    official_splits = {video: split for split, videos in split_map.items() for video in videos}
    external_sources = _personpath_external_sources(publisher_script)
    records = []
    output = output.resolve()
    with ZipFile(amodal_path) as annotations, ZipFile(detector_path) as detector:
        complete = set(detector.namelist())
        missing = [Path(video).stem for video in official_splits
                   if f'mot_format/{Path(video).stem}.txt' not in complete]
        if missing:
            raise ValueError(f'Missing {len(missing)} publisher detections: {", ".join(missing[:5])}')
        for video, official in sorted(official_splits.items()):
            name = Path(video).stem
            det_member = f'mot_format/{name}.txt'
            ann_member = f'anno_amodal_2022/{video}.json'
            annotation_bytes = annotations.read(ann_member)
            annotation = json.loads(annotation_bytes)
            info = annotation['metadata']
            source_length = int(info['number_of_frames'])
            source_fps = info['fps']
            with detector.open(det_member) as source, TextIOWrapper(source, encoding='utf-8') as lines:
                keyframes, labels, boxes = personpath_rows(annotation['entities'], lines, source_length)
            source_id, collection, asset = external_sources.get(
                video, (f'publisher-asset:{name}', 'publisher-asset', video))
            split = 'exploratory_external' if video in external_sources else (
                'test' if official == 'test' else 'development' if int(
                    hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) % 5 == 0 else 'calibration')
            home = output / name
            home.mkdir(parents=True, exist_ok=True)
            gt = home / 'gt.txt'
            detections = home / 'detections.txt'
            _write_mot(gt, labels)
            _write_mot(detections, boxes)
            records.append({
                'name': name, 'source_id': source_id, 'source_collection': collection,
                'source_asset': asset, 'split': split, 'official_split': official,
                'frames': len(keyframes), 'fps': 5, 'source_frames': source_length,
                'source_fps': source_fps, 'keyframe_source_frames': keyframes,
                'gt': str(gt), 'detections': str(detections), 'gt_sha256': sha256(gt),
                'detections_sha256': sha256(detections), 'gt_rows': len(labels),
                'detection_rows': len(boxes), 'annotation_member': ann_member,
                'annotation_member_sha256': hashlib.sha256(annotation_bytes).hexdigest(),
                'detector_member': det_member, 'detector_zip_crc32': f'{detector.getinfo(det_member).CRC:08x}',
                'ignored_classes': [8, 13],
            })
    return {
        'dataset': 'PersonPath22', 'official_source': PERSONPATH_SOURCE,
        'licenses': 'CC BY-NC 4.0 for the released dataset',
        'ground_truth': f'{PERSONPATH_ANNOTATIONS}anno_amodal.zip',
        'ground_truth_sha256': sha256(amodal_path),
        'detections': PERSONPATH_DETECTOR, 'detections_archive_sha256': sha256(detector_path),
        'detector_provenance': 'Publisher-released FCOS person detections, trained on CrowdHuman train and validation.',
        'official_split_source': f'{PERSONPATH_ANNOTATIONS}splits.json',
        'official_split_sha256': sha256(splits_path),
        'timebase': 'Both MOT files are one-based over manually annotated keyframes at nominal 5 Hz. '
                    'keyframe_source_frames gives exact one-based raw video frame alignment.',
        'box_format': 'Native source pixel xywh coordinates, unchanged.',
        'annotation_rules': 'Normal/standing/sitting people are class 1, labelled distractors class 8, and crowds class 13.',
        'ignored_classes': [8, 13], 'source_groups_verified': False,
        'source_id_scope': 'Known remappings from publisher download script; other released assets have distinct IDs '
                           'but original recording and site independence is unverified.',
        'publisher_mapping_sha256': sha256(publisher_script),
        'split_rule': 'Known external-access assets are exploratory. Other official test assets are test; '
                      'the remaining official train assets are development when the first 32 SHA-256 bits of '
                      'source_id are divisible by five, and calibration otherwise.',
        'expected_sequence_count': len(official_splits), 'missing_sequences': [], 'sequences': records,
    }


def bdd_rows(frames, name):
    """Export BDD pedestrian truth and ignore boxes onto the shared MOT pixel convention."""
    rows = []
    for index, frame in enumerate(frames):
        if frame['index'] != index or frame['name'] != f'{name}-{index + 1:07d}.jpg':
            raise ValueError(f'Unaligned BDD annotation: {frame["name"]}')
        for annotation in frame['labels']:
            category = annotation['category']
            if category not in {'pedestrian', 'rider', 'other person'}:
                continue
            box = annotation['box2d']
            left, top, right, bottom = (box[corner] for corner in ('x1', 'y1', 'x2', 'y2'))
            crowd = annotation.get('attributes', {}).get('Crowd', False)
            kind = 13 if crowd else 1 if category == 'pedestrian' else 8
            if right > left and bottom > top:
                rows.append((index + 1, int(annotation['id']), left + 1, top + 1,
                             right - left, bottom - top, 1, kind, -1, -1))
    return rows


def bdd_pilot(dataset, output):
    """Generate frozen COCO-only person detections on the recorded BDD pilot selection."""
    import numpy as np
    import torch
    import torchvision
    from scripts.detect_mot17_dev import (DATA, MIN_SCORE, NMS_IOU, SIZE, WEIGHTS_URL,
                                          YOLOX_COMMIT, load_image, load_yolox, person_boxes,
                                          write_boxes)

    image_manifest_path = dataset / 'image-source-manifest.json'
    image_manifest = json.loads(image_manifest_path.read_text(encoding='utf-8'))
    weight_digest = sha256(DATA / 'yolox_x.pth')
    if weight_digest != COCO_WEIGHT_SHA256:
        raise ValueError('Unexpected COCO detector checkpoint checksum')
    model, preproc = load_yolox()
    model.half()
    records = []
    output = output.resolve()
    for source in image_manifest['sequences']:
        name = source['name']
        annotation = dataset / 'labels' / 'val' / f'{name}.json'
        frames = json.loads(annotation.read_text(encoding='utf-8'))
        labels = bdd_rows(frames, name)
        if source['frames'] != len(frames):
            raise ValueError(f'BDD label/image length mismatch: {name}')
        images = dataset / 'val' / name / 'img1'
        home = output / name
        home.mkdir(parents=True, exist_ok=True)
        labels_path, cache = home / 'gt.txt', home / 'detections.txt'
        _write_mot(labels_path, labels)
        digest = hashlib.sha256()
        staging = cache.with_suffix('.tmp')
        start = perf_counter()
        batch_size = 8
        count = 0
        with staging.open('w', newline='', encoding='utf-8') as handle, torch.inference_mode():
            writer = csv.writer(handle, lineterminator='\n')
            for frame_ids in batched(range(1, len(frames) + 1), batch_size):
                arrays, geometry = [], []
                for index in frame_ids:
                    image = images / f'{index:06d}.jpg'
                    digest.update(bytes.fromhex(sha256(image)))
                    pixels = load_image(image)
                    height, width = pixels.shape[:2]
                    array, ratio = preproc(pixels, SIZE)
                    arrays.append(array)
                    geometry.append((ratio, width, height))
                predictions = model(torch.from_numpy(np.stack(arrays)).to('cuda', torch.float16)).float()
                for index, prediction, dimensions in zip(frame_ids, predictions, geometry, strict=True):
                    count += write_boxes(writer, index, person_boxes(prediction, *dimensions))
        if digest.hexdigest() != source['frame_sha256']:
            raise ValueError(f'BDD frame checksum mismatch: {name}')
        staging.replace(cache)
        seconds = perf_counter() - start
        records.append({
            'name': name, 'source_id': name, 'split': 'test', 'official_split': 'val',
            'frames': len(frames), 'fps': 5, 'gt': str(labels_path), 'detections': str(cache),
            'gt_sha256': sha256(labels_path), 'detections_sha256': sha256(cache),
            'official_annotation_sha256': sha256(annotation), 'public_images': source,
            'ignored_classes': [8, 13], 'gt_rows': len(labels), 'detection_rows': count,
            'cache_wall_seconds': seconds,
        })
        print(f'{name}: {len(frames)} frames, {count} detections, {seconds:.1f} seconds', flush=True)
    return {
        'dataset': 'BDD100K MOT', 'ground_truth_source': BDD_LABEL_SOURCE,
        'images_source': image_manifest['source'], 'selection': image_manifest['selection'],
        'source_groups_verified': False,
        'source_id_scope': 'Official video IDs; original recording and site overlap are unresolved.',
        'protocol': 'Exploratory, enriched independent-domain pilot; no cross-domain certificate.',
        'box_format': 'One-based frame index and pixel x, y; unmodified box widths and heights.',
        'detector': 'YOLOX-X COCO', 'detector_checkpoint': WEIGHTS_URL, 'weights_sha256': weight_digest,
        'detector_commit': YOLOX_COMMIT, 'input_size_hw': SIZE, 'precision': 'float16',
        'person_score_floor': MIN_SCORE, 'nms_iou': NMS_IOU, 'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'expected_sequence_count': len(records), 'sequences': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--cache', type=Path, action='append')
    parser.add_argument('--split', choices=('train', 'val'), action='append')
    parser.add_argument('--personpath-downloads', type=Path)
    parser.add_argument('--personpath-output', type=Path)
    parser.add_argument('--bdd-pilot-data', type=Path)
    parser.add_argument('--bdd-pilot-output', type=Path)
    parser.add_argument('--partial', action='store_true', help='allow incomplete DanceTrack cache coverage')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.personpath_downloads and args.personpath_output:
        manifest = personpath_manifest(args.personpath_downloads, args.personpath_output)
    elif args.bdd_pilot_data and args.bdd_pilot_output:
        manifest = bdd_pilot(args.bdd_pilot_data, args.bdd_pilot_output)
    elif args.dataset and args.cache and args.split:
        manifest = dance_manifest(args.dataset, args.cache, splits=args.split, partial=args.partial)
    else:
        parser.error('provide DanceTrack, PersonPath22, or BDD100K pilot inputs')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = args.out.with_suffix(args.out.suffix + '.tmp')
    staged.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    staged.replace(args.out)
    print(f'Verified {len(manifest["sequences"])}/{manifest["expected_sequence_count"]} sequences')


if __name__ == '__main__':
    main()
