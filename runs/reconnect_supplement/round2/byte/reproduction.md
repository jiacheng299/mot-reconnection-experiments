# ByteTrack acceptance-matched comparison

Run from the workspace root with the project virtual environment.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& .venv/Scripts/python.exe -m reconnect_study.supplement_matched --tracker byte --workers 6
& .venv/Scripts/python.exe -m reconnect_study.supplement_matched_audit --tracker byte

$manifest = 'runs/reconnect_study/personpath22/manifest.json'
foreach ($cap in @(6, 12, 24)) {
    $dir = "runs/reconnect_supplement/round2/byte/B$cap"
    & .venv/Scripts/python.exe -m reconnect_study.supplement_matched_eval "$dir/test.json" "$dir/locked.json" "$dir/report.json" --run-manifest $manifest --run-config "$dir/config.json" --trace-directory "$dir/traces"
}
& .venv/Scripts/python.exe -m reconnect_study.supplement_matched_audit --tracker byte --test
& .venv/Scripts/python.exe -m reconnect_study.supplement_matched_verify --workers 6
```

The first command reads the pinned manifest and checks its annotation and detection file hashes. It reconstructs the original threshold values from the exact archived development candidates bound to the frozen configuration, then verifies that these values agree exactly. Each cap's original calibration precedes its optional dense sweep. Quantile positions from 0.35 through 0.97 are inclusive at 0.01 increments. The original strategy IDs are preserved, with the new thresholds appended by score.

Within one scoring family, all thresholds yielding the same sequence of live policy decisions yield the same deterministic tracker states. A full replay records the below-cap score constraints. Accepting a finite score gives an inclusive upper bound for the cutoff. Rejecting a finite score gives an exclusive lower bound. Once the accepted action budget is saturated, later proposals do not constrain the cutoff. Only thresholds within the resulting interval share the representative's trajectory. For each source and family where sharing occurs, an additional policy cutoff is independently replayed with the regular gate, then compared for equality of complete trajectories and decisions as well as evaluation records.

The raw assignment score is the negated vendor association cost. The additive and structured scores are dimensionless learned scores. The gate accepts scores at or above the cutoff. The matched quantity uses evaluable accepted actions per source, and the runtime cap applies to all accepted actions including those subsequently ignored by the evaluator.

The first audit command compares the new B=12 original grid with archived per-video calibration counts. The second checks every emitted test trace and compares unchanged reference replays to their archived results. The final command independently checks paired intervals from the saved per-video counts and computes HOTA from the saved output tracks. The outer workspace lacks a resolvable Git HEAD, so the input inventory binds the current replay and evaluation source files by SHA-256. The audit records vendor commits and working-tree status. This analysis uses the archived 74 official-test videos as an exploratory sensitivity check because the 55 externally mapped assets lack verification as independent recordings.
