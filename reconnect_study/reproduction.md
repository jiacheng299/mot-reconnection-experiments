# Reproduce the manuscript experiment

Run the commands from the workspace root. The existing inputs are pinned in
`runs/reconnect_study/personpath22/manifest.json`. Dataset files are public
PersonPath22 releases. Respect the publisher's CC BY-NC terms for data use.

```powershell
$python = '.venv/Scripts/python.exe'
$p = 'runs/reconnect_study/personpath22'
$env:OMP_NUM_THREADS = '2'

& $python -m unittest discover -s reconnect_study/tests -v
& $python -m reconnect_study.study replay "$p/manifest.json" "$p/byte-config.json" calibration "$p/recheck-calibration-byte.json"
& $python -m reconnect_study.study replay "$p/manifest.json" "$p/byte-config.json" test "$p/recheck-test-byte.json" --locked-policies "$p/locked-byte.json" --overall-protocol personpath22-native-keyframe
& $python -m reconnect_study.study replay "$p/manifest.json" "$p/bot-config.json" test "$p/recheck-test-bot.json" --locked-policies "$p/locked-bot.json" --overall-protocol personpath22-native-keyframe
& $python -m reconnect_study.certification --input "$p/finite-certificate-input.json" --output "$p/recheck-certificate.json" --alpha .20 --delta .05 --test betting --min-active-coverage .20
```

Use the released trained configurations when rechecking the finite-panel
calculation. Their exact hashes were bound to the random draw. To fit scores
again on an RTX 3060 Ti, run `reconnect_study.replay --split development`
and then `reconnect_study.study freeze` with
`--device cuda --acceptance-cap 12 --quantiles 0.35 0.60 0.80 0.93 0.97`.
Record the output as a new run. The new timing fields naturally produce a
different configuration hash.

The randomized-panel input lists the 91 prospectively drawn sources. It
combines frozen policy replays from two disjoint source lists. The separate
official-test selection depends only on the original calibration JSON. The
verified active-video bound and the official-test effect intervals use
different estimation procedures. The two analyses share some source videos.

Test traces contain both reactivation proposals and final tracks. The
recorded checksums allow direct lookup by policy and video. Ignore regions
are only used when evaluating traces. Present empirical failure counts with
their admitted-action denominator and natural-gap recovery with its
separate, fixed denominator. The effect intervals resample whole source
videos, and the bound uses the recorded exact uniform design.
