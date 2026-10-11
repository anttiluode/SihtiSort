# Protocol 6 — register each spike before averaging

Written 11 October 2026, before any code for this addendum. Thresholds are set here. Code may be debugged on seeds 92–94 only; the fresh sets below are run once on frozen code.

## Why (Sol)

In Protocol 5, frame correction failed under sine drift even with the true drift. The code averaged a part's spikes first and then subtracted the mean drift from the averaged footprint's centroid. A fast-moving probe has already blurred that average, so the oracle tested the wrong order of operations. Current sorters move each spike back into place *before* combining them.

Sol also pointed out that the switch comparison's "the probe has barely moved" was never measured: 15 spikes on each side can span several seconds.

## Registration

Each spike's 16-channel snippet is shifted along the channel axis by minus the drift at its own time, using linear interpolation between neighbouring channels; at the probe edges, the edge channel is repeated. Registered spikes are then averaged, and the centroid is compared with no further correction.

## Methods (same sorting for all)

| Method | |
|---|---|
| frame-oracle-centroid | Protocol 5: average, then subtract the true drift from the centroid |
| frame-oracle-registered | register each spike by the true drift, then average |
| frame-estimated-centroid | Protocol 5, with the blind population drift estimate |
| frame-estimated-registered | register each spike by the estimated drift, then average |
| frame-wrong-registered | register by the estimate reversed in time (control) |
| switch | Protocol 5's switch comparison, now recording the time span of every comparison |

The thresholds, the rate refuter and the drift estimator are all unchanged from Protocol 5.

## Fresh sets (12 recordings each, same generator settings as Protocol 5)

- linear drift: seeds 170–181
- sine drift: seeds 190–201
- no drift: seeds 210–221

## Gates

A method **passes a set** with the same rule as Protocol 5: ≤ 1 false reopening, mood drop ≤ 1, stable drop ≤ 2, and ≥ 70% of separable merged pairs apart (not applicable if a set has none).

- **G1 — mechanism.** frame-oracle-registered passes the sine set *and* makes fewer false reopenings there than frame-oracle-centroid. If so, the Protocol 5 sine failure came from averaging before registering.
- **G2 — practical.** frame-estimated-registered passes all three sets.
- **G3 — control.** frame-wrong-registered makes ≥ 3 false reopenings on the linear set.
- **Reported, no gate:** for the switch comparison, the distribution of the time span covered by each comparison (first spike before the switch to last spike after it), per set.
