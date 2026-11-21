# Statistical method for closed-loop reconnection

## Outcomes and policy family

For each source video \(s\) and frozen reconnect policy \(j\), let \(W_{sj}\)
be the number of wrong or unverifiable evaluated reactivations, and let
\(A_{sj}\) be the number of evaluated accepted reactivations. Target annotations
and ignore regions enter retrospective evaluation. The active gate consumes
only tracking state and detector inputs. The source budget applies to all
accepted reactivations before evaluation, so \(0\le W_{sj}\le A_{sj}\le B\).
All replays start at the beginning of the video. This includes the effect of
early decisions on subsequent proposals and makes the outcome a property of
the entire fixed policy.

On a finite panel \(\mathcal S\), the event-weighted risk is
\[
R_j=\frac{\sum_{s\in\mathcal S} W_{sj}}{\sum_{s\in\mathcal S} A_{sj}},
\qquad \sum_{s\in\mathcal S} A_{sj}>0 .
\]
An additional active-video estimand weights equally those source videos with
at least one evaluated accept. The experimental reports show how many videos
are active, how many actions were accepted, and how many gaps retained their
original identity on the return keyframe.

## Fixed-size random calibration panel

Let \(I\) be an \(n\)-element simple random subset of \(\mathcal S\), drawn
uniformly without replacement after development and policy-family freezing.
At target risk \(r\in(0,1)\), define
\[
f_{sj,t}(r)=1+t\frac{r A_{sj}-W_{sj}}{B(1-r)},\qquad
E_{j,t}(r)=\prod_{s\in I} f_{sj,t}(r), \qquad t\in[0,1].
\]
Each factor is nonnegative. Under \(H_0:R_j\geq r\), its average over
\(\mathcal S\) is at most one. The expectation over the random draw is a
normalized elementary symmetric mean. Maclaurin's inequality gives
\[
\mathbb E_I E_{j,t}(r)
= \frac{e_n(f_{1j,t},\ldots,f_{|\mathcal S|j,t})}{\binom{|\mathcal S|}{n}}
\leq \left(\frac{1}{|\mathcal S|}\sum_{s\in\mathcal S} f_{sj,t}\right)^n
\leq 1.
\]
The fixed mixture averages the products at \(t=1/2\) and \(t=1\). Markov's
inequality converts it to a one-sided test. Bonferroni allocation over the
frozen family gives simultaneous validity across strategies sharing calibration
videos. The factor derivative is
\(t(A_{sj}-W_{sj})/[B(1-r)^2]\geq0\). Inversion therefore yields an upper
confidence bound. In the active-video analysis we condition on the number
of active sampled sources and apply the same construction to one fractional
loss \(W_{sj}/A_{sj}\) per active source. The two estimands receive separate
parts of the total error budget.

These arguments condition on the entire fixed collection of counterfactual
source-policy outcomes. Temporal dependence within each video and dependence
among different video outcomes are allowed. The randomness is the exact
uniform calibration draw. A distributional claim about future sources would
require a separate source sampling model. The official-test replays estimate
utility under their pre-existing split. The randomized panel answers the
finite-pool risk question with a separate selector.

## Related statistical work

- Angelopoulos et al., *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control*. https://arxiv.org/abs/2110.01052
- Waudby-Smith and Ramdas, *Confidence Sequences for Sampling without Replacement*. https://proceedings.neurips.cc/paper/2020/hash/e96c7de8f6390b1e6c71556e4e0a4959-Abstract.html
- Waudby-Smith and Ramdas, *Estimating Means of Bounded Random Variables by Betting*. https://arxiv.org/abs/2010.09686
