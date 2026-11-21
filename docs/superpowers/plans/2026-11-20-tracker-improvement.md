# Tracker improvement implementation plan

> Execute the unchecked steps in order and verify each before interpreting scores.

**Goal** Establish a new tracker with reproducible accuracy gains against the unmodified host and competitive same-input comparisons.

**Architecture** Reuse the frozen BoT-SORT implementation. The first-assignment reranking ran as a control. The candidate combines appearance-assisted low-score matching with cautious reactivation of recently lost tracks. The runner shares image embeddings and camera warps across configurations and independently evolves each tracker's state.

**Input scope** First compare on MOT17-10 and DanceTrack val. Existing P6 results establish the historical reference. Use a fresh, independent evaluation for a confirmatory result once a candidate is locked.

- [x] Test that the first-stage intervention preserves forbidden edges and isolated decisions. Test that the recovery path protects active tracks and rejects near ties.
- [x] Run the MOT17-10 and three-video DanceTrack development pilots. Report HOTA, AssA, IDF1, switches, and the first-stage ranking and low-confidence counters. Validate baseline tracks against P6.
- [x] Analyze the same sequences using a fixed shared detector-defined cohort. Keep truth downstream of tracking and record denominators.
- [x] Evaluate the frozen RecentRecovery candidate on the other 22 DanceTrack validation videos. Compare against BoT-SORT, OC-SORT, and TrackTrack on the same cohort. Report LowAppearance and RecoveryOnly as mechanism controls.
- [ ] Lock any method supported by development evidence, identify an independent annotated benchmark, and verify against strong same-input baselines. Check closest prior work and document where the mechanism differs before making a novelty claim.

## Locked transfer within DanceTrack development

The main outcome is the rate of correct identity continuations on `gap_low`. A confirmed person has a matched detection above 0.60, remains annotated during 1–3 frames without a matched detector box, and reappears with a matched box of score 0.10–0.60. Detector-to-annotation pairing uses one-to-one IoU matches of at least 0.50 before reading any tracker output. The numerator is the number of instances in which the tracker output at the first return frame uses its pre-gap ID. Count the same denominator for every method and show missing anchors, missed returns, and wrong IDs separately. Continuous low-score episodes and high-score returns are secondary checks. Report aggregate HOTA, AssA, IDF1, and identity switches for the same 22 videos.

The three pilot sequences are dancetrack0041, dancetrack0047, and dancetrack0097. Other official validation sequences were excluded from scoring any new candidate before locking this analysis. No candidate parameters may be changed after viewing the 22-video outcomes for this iteration.
