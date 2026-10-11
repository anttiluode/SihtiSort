# Protocol 5 — correct the frame, or compare at the switch

Written 11 October 2026, before any code for this addendum. Thresholds are set here. Code may be debugged on seeds 92–94 only; the three fresh sets below are run once on frozen code.

## Why

Protocol 4 failed D1: comparing the two parts of a neuron inside fixed 15 s windows cut false reopenings from 40 to 5, but did not remove them. The probe still moves 0.15 channels within one window, which equals the threshold. Two ways out:

- **Correct the frame (Sol).** The probe's drift is shared by every neuron. Estimate it from the whole population and subtract it before comparing positions. Three versions: the **true** drift from the simulator (an upper bound), the drift **estimated** blind from the sorted population, and a **wrong** drift (the estimate run backwards in time), which should fail if the correction is doing the work.
- **Compare at the switch (structure-yoked).** Norman-Haignere et al. (Nature Neuroscience 2025) found that auditory cortex integrates over windows of fixed absolute time, not over speech structure. Protocol 4's window was time-yoked in the same way and failed for that reason. The structure-yoked alternative anchors the comparison to events: at every point where the time-ordered union of the two parts changes label, compare the last ≤ 15 spikes before it with the first ≤ 15 after it (≥ 6 each). Across that instant the probe has hardly moved.

## Methods, all computed from the same sorting

| Method | Position comparison |
|---|---|
| naive | whole parts (Protocol 3) |
| window | shared 15 s windows (Protocol 4) |
| frame-oracle | whole parts, positions corrected by the true drift |
| frame-estimated | whole parts, positions corrected by the drift estimated from all committed addresses |
| frame-wrong | whole parts, positions corrected by the estimated drift reversed in time |
| switch | switch-anchored |
| switch+frame | switch-anchored, with the estimated correction |

**Drift estimate.** Every committed address with ≥ 6 spikes in each of ≥ 3 windows of 5 s gets a position trajectory: the centroid of its averaged spike in each window, minus its own median. The drift estimate is the median of these trajectories across addresses in each window, interpolated in time. It uses no ground truth. It is computed after the committed sorting and before any remainder decision, so it protects the reopen decisions and not the earlier merges.

Thresholds are unchanged: reopen if the gap is ≥ 0.15 channels with z ≥ 3, or if the rate refuter fires. The rate refuter is the same in every method.

## Fresh sets (12 recordings each; neurons ≥ 1 channel apart, turn-taking pair 0–0.5 channels apart)

- **linear drift**, seeds 110–121: 0.6 channels over 60 s, mean dwell 20 s (as in Protocol 4)
- **sine drift**, seeds 130–141: the probe oscillates with amplitude 0.3 channels and period 40 s, random phase, mean dwell 20 s
- **no drift**, seeds 150–161: mean dwell 8 s (as in Protocol 3)

## Gates

A method **passes a set** if, against commit mode, it has ≤ 1 false reopening (both parts dominated by the same true neuron), mood neurons kept as one unit drop by ≤ 1, stable neurons drop by ≤ 2, and ≥ 70% of separable merged pairs (offset ≥ 0.25 channels or rate ratio ≥ 1.5) end apart. If a set has no separable merged pair, the last condition is not applicable.

- **F1:** frame-estimated passes all three sets.
- **F2:** switch passes all three sets.
- **F3 (control):** frame-wrong makes ≥ 3 false reopenings on the linear set. If it doesn't, the correction is not what is doing the work.
- **F4 (bound):** frame-oracle passes the linear and sine sets. If even the true drift cannot pass, frame correction is not enough.

Every method's numbers are reported, whatever the gates say.
