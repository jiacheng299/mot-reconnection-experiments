"""Read a frozen manifest and matched MOT-format annotation/detection inputs."""

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath


@dataclass(frozen=True)
class Sequence:
    name: str
    source_id: str
    frames: int
    fps: float
    gt: Path
    detections: Path
    split: str
    detection_origin: int = 1
    gt_origin: int = 1
    minimum_visibility: float | None = None
    ignored_classes: tuple = ()


def boxes(path, *, truth, origin=1, minimum_visibility=None):
    by_frame = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            frame, identity = map(int, row[:2])
            x, y, w, h = map(float, row[2:6])
            score = float(row[6])
            if w <= 0 or h <= 0 or frame < 1:
                raise ValueError(f"Invalid box or frame in {path}: {row}")
            x, y = x - origin, y - origin
            box = (x, y, x + w, y + h)
            if truth:
                if score != 1 or (len(row) > 7 and float(row[7]) != 1):
                    continue
                if (minimum_visibility is not None and len(row) > 8
                        and float(row[8]) < minimum_visibility):
                    continue
                people = by_frame.setdefault(frame, {})
                if identity in people:
                    raise ValueError(f"Duplicate identity {identity} in {path} at frame {frame}")
                people[identity] = box
            else:
                by_frame.setdefault(frame, []).append((*box, score))
    return by_frame


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ignored_boxes(path, classes, *, origin=1):
    by_frame = {}
    if not classes:
        return by_frame
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 8 or int(float(row[7])) not in classes:
                continue
            x, y, w, h = map(float, row[2:6])
            x, y = x - origin, y - origin
            if w > 0 and h > 0:
                by_frame.setdefault(int(row[0]), []).append((x, y, x + w, y + h))
    return by_frame


def archived_path(directory, location):
    """Resolve historic workspace paths without modifying frozen manifests."""
    raw = Path(location)
    windows = PureWindowsPath(location)
    if not raw.is_absolute() and not windows.is_absolute():
        return directory / raw
    parts = windows.parts if windows.is_absolute() else raw.parts
    anchors = [i for i, item in enumerate(parts[:-1]) if item in {"data", "runs"}]
    if anchors:
        root = Path(os.environ.get("MOT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[1]))
        relocated = root.joinpath(*parts[anchors[-1]:])
        if relocated.exists() or not raw.exists():
            return relocated
    return raw


def read_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_bytes())
    rows = payload["sequences"] if isinstance(payload, dict) else payload
    sequences = []
    names = set()
    for row in rows:
        name = row.get("name", row.get("sequence"))
        if not name or name in names:
            raise ValueError(f"Missing or duplicate sequence name: {name}")
        names.add(name)
        paths = {key: archived_path(path.parent, row[key]) for key in ("gt", "detections")}
        for key, value in paths.items():
            expected = row.get(f"{key}_sha256")
            if expected is not None and digest(value) != expected:
                raise ValueError(f"Content changed: {value}")
        sequences.append(Sequence(name, row["source_id"], int(row["frames"]), float(row["fps"]),
                                  paths["gt"], paths["detections"], row["split"],
                                  row.get("detection_origin", 1), row.get("gt_origin", 1),
                                  row.get("minimum_visibility"), tuple(row.get("ignored_classes", ()))))
    source_splits = {}
    for sequence in sequences:
        previous = source_splits.setdefault(sequence.source_id, sequence.split)
        if previous != sequence.split:
            raise ValueError(f"Source {sequence.source_id} appears in both {previous} and {sequence.split}")
    return sequences
