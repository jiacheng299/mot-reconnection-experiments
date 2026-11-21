# Recovery decision and independent checks

The main event keeps the round-one definition. A detection above 0.60 is followed by one to three frames without an annotation-matched box and then a box scoring at most 0.60. The person remains annotated throughout. Correct means retaining the pre-gap tracking ID.

The new decision leaves active-track costs intact. For unmatched recent lost tracks, an additive risk score combines box displacement, appearance distance, detection confidence and time since the last update. A close competing identity or detection prevents reactivation. Compare the four fixed cutoffs 0.28, 0.37, 0.46, 0.55 on MOT17-10 and the original three DanceTrack pilot sequences. Report correct, wrong, and missed events plus overall HOTA and IDSW. Select the development cutoff maximizing pooled correct minus wrong events. Break ties by higher correct count and then the stricter cutoff.

Before opening results for the selected method on MOT17-11/13, write its threshold and source hash. Keep these sequences out of selection. Report the locked method next to the previous recovery variant, BoT-SORT and archived TrackTrack under matching detector inputs. These sequences were used in the original project benchmark, and have never been used to tune this recovery decision.

For comparison at a fixed recovery count, report observed settings that reach at least the locked method's correct count, with the actual correct and wrong counts for each. Label exact-count matches where they exist. Sweep TrackTrack's additive low-confidence penalty at 0.12, 0.20 (P6 reference), 0.28, 0.36, 0.44 and 0.52. Keep its other settings fixed and require byte equality at 0.20. The held-out sweep is a comparison and does not select the locked method.
