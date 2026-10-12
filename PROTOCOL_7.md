# Protocol 7 — make the forced split pay for how it was generated

Written 12 October 2026, before any code for this addendum. Thresholds are set here. Code may be debugged on seeds 92–94 only; the fresh sets below are run once on frozen code.

## Why (Sol)

The forced split is a procedure built to find two groups. Under a one-neuron truth it still finds groups that differ, especially when the probe drifts. The diagnostic in `results/null_split.json` showed this: with the true drift removed, about 10% of small one-neuron splits still crossed the 0.15-channel bar, and z ≥ 3 appeared in 18–69% of null trials. A fixed threshold plus a bootstrap z-score does not account for how the split was generated.

## The calibrated null

For every forced-split candidate (a one-state address with ≥ 60 spikes), the sorter builds a one-neuron world for *that* address and runs the identical procedure on it:

1. Register each of the address's purified spikes by the estimated drift, and average them into one stationary template.
2. Place that template back at each spike's own time and drifted position, scaled by the spike's own fitted amplitude.
3. Add noise by resampling the address's own residuals (each purified spike minus its fitted template), with replacement.
4. Run the exact forced split (fit on even spikes, assign odd spikes), the same drift handling and the same gap measurement. Do this **30 times**.

Decision for a forced split:

- **Position** fires only if the observed gap is ≥ max(0.15, μ + 3σ) of the 30 null gaps, *and* z ≥ 3 as before.
- **Rate** fires only if the rate refuter fires as before *and* its z is ≥ μ + 3σ of the null rate z-scores. Null draws in which the parts do not take turns count as z = 0.

Two-state addresses (the "states" remainder) are unchanged; they were 2 of the 44 false reopenings in Protocol 6.

## Variants (same sorting for all)

| Variant | |
|---|---|
| uncal-registered | Protocol 6's frame-estimated-registered, uncalibrated |
| cal-registered | the same, with the calibrated null for forced splits |
| cal-uncorrected | calibrated null, but no drift correction at all; the null itself contains the measured drift |

## Fresh sets (larger: 20 recordings each, same generator settings as Protocols 5–6)

- linear drift: seeds 230–249
- sine drift: seeds 250–269
- no drift: seeds 270–289

The recording is the resampling unit. 95% intervals are bootstrapped over recordings (2,000 resamples).

## Gates

- **C1 — safe.** cal-registered makes ≤ 2 false reopenings in each set of 20; mood neurons drop ≤ 1 and stable neurons ≤ 2 per set, against commit mode.
- **C2 — calibration is what helps.** Pooled over the 60 recordings, cal-registered has fewer false reopenings than uncal-registered, and the 95% recording-bootstrap interval of the per-recording difference excludes 0.
- **C3 — still finds real pairs.** Pooled over the three sets, cal-registered separates ≥ 70% of the separable false merges (offset ≥ 0.25 channels or rate ratio ≥ 1.5) that commit mode leaves merged.
- **Reported:** cal-uncorrected on the same measures, and the separable-pair recall of all three variants with recording-bootstrap intervals.

## Deviation found on the calibration seeds (before the fresh run)

Step 3 said to resample residuals *with replacement*. On seed 92 that made the null measure the wrong thing: a duplicated outlier residual added to near-identical templates forms a tight little cluster, and the two-way mixture split off 5–30 such spikes in most draws (a part too small to test), instead of splitting the neuron along the drift as it does in the real data. Each draw now uses every residual exactly once, in a random order (a permutation). Nothing else changed. Draws in which a part is too small to test count as gap 0, as in the real procedure, which then stays silent.
