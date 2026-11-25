# Reproduce the third-round BoostTrack-v1 component comparison

Run the commands from a checkout containing the frozen PersonPath22 inputs and pinned vendor sources. Use a separate working copy when generating results. The test replay refuses to replace its archived `test.json`. Preserve the supplied locks and use them directly for an audit of the recorded experiment.

```powershell
python -m reconnect_study.round3_boost_audit
python -m reconnect_study.round3_boost_audit --host bot --baseline runs/reconnect_study/personpath22/development-bot.json --output runs/reconnect_supplement/round3/boost/bot/development-numeric-audit.json
python -m reconnect_study.round3_boost_calibrate --workers 6
python -m reconnect_study.round3_boost_bot_calibrate --workers 10
python -m reconnect_study.round3_boost_eval --host byte
python -m reconnect_study.round3_boost_eval --host bot
python -m reconnect_study.round3_summary
python -m reconnect_study.round3_figure
python -m reconnect_study.round3_boost_verify --workers 8
```

The numeric audit imports `tracker/assoc.py` from BoostTrack commit `fb5bfc3a8f067476565e753b3a73df4d757c9d03`, placed at `runs/reconnect_supplement/round3/boost/upstream/BoostTrack` or supplied with `--upstream`. Its author reference and local adapter receive the same host-derived inputs, including identical Mahalanobis distances. Score equivalence therefore checks the score formula and ranking for those inputs. ByteTrack reorders its existing `xyah` state to `xyhr`. Motion-only BoT-SORT uses `xywh` and maps aspect ratio to `w/h`, with first-order covariance propagation `J Σ Jᵀ`. The variance of `w/h` is approximated by this propagation. The formulas retain the author's v1 shape term, confidence weighting and column-normalized Mahalanobis similarity. Neither host uses BoostTrack detection boosting, prediction, or assignment. High scores favor accepting a host-proposed lost identity under the per-source cap of 12.

The archived ByteTrack sweep was executed directly on the dense grid. A calibration-only check of the original five thresholds finds that the high setting misses its target by 0.632 accepts per source, above its 0.322 tolerance, so the prespecified refinement would be triggered. ByteTrack's existing lock was calibrated in one pass over the dense 0.35–0.97 quantile grid. BoT-SORT used the original grid first and expanded to the dense grid for the unmatched points before freezing all three selections. Calibration ties prefer lower volume, then a higher cutoff, then the policy ID. Each selected policy replays independently from the first frame. The test sample consists of 74 previously analyzed PersonPath22 videos, so the result is exploratory. Test acceptance-volume closeness is reported separately from jointly attaining the calibration target.

The portable package includes configuration, calibration and lock files, per-video test counts and trace hashes, merged tables, and the figure. Test replay regenerates the full trace files locally. The final audit reads those trace files to recompute HOTA and recomputes the paired bootstrap intervals from per-video counts. In a compact checkout, `round3_summary` and `round3_figure` can rebuild the tables and figure directly from the archived results without the large traces or detection inputs.
