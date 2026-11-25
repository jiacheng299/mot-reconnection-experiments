# Reproduce the 12/15/23-feature comparison

Run from the workspace root. The old B=12 results are reused after their manifest, configuration, split and digest bindings are verified. The ByteTrack high-point event counts are checked against the archived 158/390, 114/393 and 98/381 anchors. New output tracks and event records are saved per policy and video.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& .venv/Scripts/python.exe -m reconnect_study.round3_ablation --tracker byte --workers 4 --device cuda
& .venv/Scripts/python.exe -m reconnect_study.round3_ablation --tracker bot --workers 2 --device cuda
& .venv/Scripts/python.exe -m reconnect_study.round3_ablation_eval --tracker byte
& .venv/Scripts/python.exe -m reconnect_study.round3_ablation_eval --tracker bot
& .venv/Scripts/python.exe -m reconnect_study.round3_ablation_finalize --workers 4
```

The 15-feature design computes the three summary features independently and checks them against the existing 23-feature prefix on every evaluable development proposal. The old scoring functions must reproduce all stored development quantile cutoffs exactly. In calibration, thresholds share a full replay only when all live scoring decisions would coincide. Each source where sharing occurs also receives a separate comparison of the full track and proposal sequences with the ordinary gate. Every new policy in the held-out run executes independently from frame one. This held-out set was previously analyzed, so results are exploratory.

The generated configurations bind the absolute Windows paths of their source code. When reproducing on another machine, resolve the corresponding local source files and verify them against the recorded SHA-256 checksums, or regenerate a configuration and lock in a separate local run using the frozen inputs. Keep the original archived lock and report intact.
