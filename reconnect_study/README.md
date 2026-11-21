# Reconnection study

This project evaluates a runtime gate for reactivating lost track identities.
The tracker receives cached detector boxes, while annotations are used to
label recovery events after tracking.

## Runtime API

Create one policy instance for each source. Carry the same instance across
clips from that source when the source budget is shared. Detector frame
indices start at 1. Each detection is `(x1, y1, x2, y2, score)` in pixels.

```python
from pathlib import Path
from reconnect_study.policy import load_policy
from reconnect_study.tracker import run_bytetrack

gate = load_policy(Path("path/to/byte-config.json"), "structured_0")
result = run_bytetrack(detections_by_frame, frame_count, policy=gate, frame_rate=5)
```

The configuration pins the feature scaling, learned coefficients,
threshold grid, tracker variant, and optional accepted-action cap. A
separate selection record identifies the policy evaluated on the test split.
Run logs expose actions and emitted tracks independently.

## Experiment organization

- `data_setup.py` prepares immutable manifests for released data and cached
  detections. The PersonPath22 workflow uses its original amodal annotations
  and aligns the publisher's FCOS rows to manual keyframes.
- `replay.py` runs an unrestricted tracker and stores development feature
  labels. `study.py freeze` uses that file to fit scores and create frozen
  candidate thresholds. `study.py replay` runs a chosen split under the
  frozen policies.
- `select_policies.py` uses calibration summaries to fix policies for the
  official test comparison. The test replay can export complete decision
  traces and TrackEval outputs.
- `finite_panel.py` combines the two source partitions for a preregistered
  random calibration draw. `certification.py` verifies the draw and the
  replay configuration, then calculates simultaneous upper confidence
  bounds. The finite-population risk guarantee is about the fixed pool.
- `overall.py` computes the TrackEval metrics. `ocsort_reference.py` runs
  the external OC-SORT reference. `effect_sizes.py` computes empirical
  paired resampling intervals at the source level.

## Verification

Run from the workspace root with the Python environment that provides
NumPy, SciPy, PyTorch, FilterPy, lap and TrackEval.

```powershell
$env:OMP_NUM_THREADS = '2'
& .venv/Scripts/python.exe -m unittest discover -s reconnect_study/tests -v
```

Experiment records are in the `runs/reconnect_study` workspace folder.
The preparation scripts preserve upstream dataset identifiers and input
digests. The reproducibility protocol records which figures and tables use
exploratory data, the official split, or the finite panel.

## Released operating points

- Official-test ByteTrack risk–recovery comparison: `structured_1` in
  `runs/reconnect_study/personpath22/byte-config.json`.
- Fixed-panel active-video certificate: `structured_3` in the same
  config. Use the action cap already saved in the file. The certificate
  applies to the fixed PersonPath22 panel and evaluation protocol.
- Official-test motion-only BoT-SORT strict comparison: `structured_2`
  in `runs/reconnect_study/personpath22/bot-config.json`.

The editable sections are `method.tex` and `experiments.tex`, with
references in `references.bib`. Compile `publication_preview.tex` from
this directory for a standalone review PDF. The experimental record
includes the table interpretation, hardware, and source split.
