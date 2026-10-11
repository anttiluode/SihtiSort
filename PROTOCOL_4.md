# Protocol 4 — the drift control

Written 11 October 2026, before any code for this addendum. Thresholds are set here. Code may be debugged on seeds 92–93 only; seeds 80–91 are run once on frozen code.

## Why (Sol's suggestion)

Protocol 3's remainder reopens an address when its two parts sit at different places on the probe. Real probes drift relative to the tissue, so one neuron's spikes from early in a recording sit somewhere else than its spikes from late in it. If a neuron also changes waveform state, its two states can occupy different stretches of time, and therefore different average positions. That is exactly the evidence the position refuter reads as "two neurons". **The correct answer is still one neuron.**

## Drift in the generator

The whole probe moves: every neuron's position shifts linearly by a total of **0.6 channels** over 60 s, in a random direction per recording. To make the confound bite, mood switches and the turn-taking blocks are slow: **mean dwell 20 s** (minimum 5 s). Each state therefore covers only a few long stretches of the recording. Otherwise the layout matches the identifiability sets: neurons ≥ 1 channel apart, and the turn-taking pair 0–0.5 channels apart.

## Drift-aware position test

Two parts can only be compared fairly at the same time. The recording is cut into **15 s windows**. In each window where both parts have ≥ **6** test spikes, each part's averaged spike gives a position, and the gap between them is taken. The drift-aware gap is the spike-weighted mean of those within-window gaps, and its standard error is bootstrapped over windows. With no shared window, the position test is silent. The thresholds are unchanged (≥ 0.15 channels and z ≥ 3), and the rate refuter is unchanged.

Both versions are computed in the same run: **naive** (Protocol 3) and **drift-aware**.

A remainder status of "agrees" is renamed **"no contradiction at tested resolution"**, following Sol: the old label overstated negative evidence.

## Gates (drift set, seeds 80–91, full 60 s recordings)

A *false reopening* is a reopening in which both parts are dominated by the same true neuron.

- **D0 — the danger is real (prediction).** The naive remainder makes ≥ 3 false reopenings, pooled. If it makes fewer, this drift does not fool it, and that is reported as the result.
- **D1 — drift-aware keeps true neurons.** ≤ 1 false reopening pooled; mood neurons recovered as one unit drop by ≤ 1, and stable neurons by ≤ 2, against commit mode.
- **D2 — drift-aware still separates.** Among turn-taking pairs that commit mode leaves merged and that are separable in principle (offset ≥ 0.25 channels or rate ratio ≥ 1.5), ≥ 70% end apart.
- **D3 — no-drift regression.** On seeds 50–61 (no drift, full recordings), the drift-aware version separates ≥ 70% of the separable merged pairs (Protocol 3's naive version: 9/9), with mood drop ≤ 1 and stable drop ≤ 2.

## Not a claim

Linear drift of the whole probe is the simplest kind. Real drift can be non-linear and can change a neuron's spike shape on top of moving it, and real neurons can change both rate and shape together. The point here is to find out whether the refuter confuses a moving source with two sources, and whether comparing at matched times fixes that.
