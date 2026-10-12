# Protocol R2 — where does the state enter? A two-site gain invariant

Written 12 October 2026, before any code for this protocol. Thresholds are set here. Code may be debugged and tuned on calibration seeds 920–922 only; the fresh seeds below are run once on a frozen commit. It reuses the neuron of Protocol R1 (`sihtisort/cable.py`, unchanged).

## Why, and a correction first

After R1 I proposed a gate: collect the neuron's state-conditioned response kernels `K(τ; x) = ∂y(t+τ)/∂u(t)` and ask whether they stay in one fixed subspace (nonlinearity *between* fixed filters) or rotate with the state (nonlinearity *inside* the operator). That gate does not separate the two classes:

- In an LNL cascade (filter, static nonlinearity, filter) the tangent kernel is `K(τ; t) = ∫ q(τ−s) g'(z(t+s)) h(s) ds`. The gain varies along the response's own future, so the kernel's *shape* changes with the state too, and the family of kernels has no fixed finite span.
- More fundamentally, Boyd & Chua (1985): every time-invariant fading-memory system can be approximated uniformly by a finite bank of linear filters followed by a static polynomial. From input-output behaviour alone, "inside" versus "between" is a question of economy (how many fixed filters it takes), not a yes/no property.

What *can* be decided from soma recordings is narrower and exact.

**The invariant.** Suppose all the state dependence between two input sites and the recording is a single pointwise gain applied after their signals merge: `y = f(v)` with `v` linear in both inputs (any static `f`). Probe both sites at the same moments with small, independent random-sign currents. For every state bin `b`,

`K₁(τ|b) / K₁(τ) = K₂(τ|b) / K₂(τ) = m_b(τ) / m(τ)`,

where `K_s(τ)` is the kernel pooled over all bins and `m_b(τ) = E[f′(v(t+τ)) | b]`. The gain cancels in the per-site ratio. Define

`Δ_b(τ) = K₁(τ|b)/K₁(τ) − K₂(τ|b)/K₂(τ)`,

which is zero in expectation under the invariant. If the state changes the paths from the two sites differently (a dendritic, pre-merge nonlinearity such as NMDA), or if anything filters the output after the gain, `Δ` departs from zero.

So the test answers: **is this neuron's state dependence a pointwise gain at the readout, or does it live somewhere else?** A rejection does *not* by itself say "inside the operator": an output nonlinearity followed by a filter also rejects. That case is included as a control so the result can be read correctly.

The question with a real answer: in R1, the cascade's residue on the NMDA neuron in ordinary activity (regime A) was clean, 0 of 10. Can **small probes in the same ordinary activity** see what the residue could not?

## Truths (same seed ⇒ same cell, synapses, input trains and noise as R1's `make_cell`)

| Truth | | invariant holds? |
|---|---|---|
| T0 passive | the cable | yes, exactly (no state dependence) |
| T1 passive + tanh readout | R1's T1 | yes, exactly in expectation |
| T1f passive + tanh readout + 8 ms low-pass | an LNL at the soma | no, though nothing is dendritic |
| T2 NMDA | R1's T2 | not required; this is the test |

## Measurement

- Regime A only (independent Poisson input), 0.5–50 s, observed at 500 Hz with OU and measurement noise as in R1.
- **Probes:** every one of the 8 input sites receives a current `ε·ξ_s(t)`, where `ξ_s(t) = ±1` is redrawn independently each 2-ms observation step. `ε` is set on the calibration seeds so that the pooled soma kernel has a peak signal-to-noise ratio of at least 10 on T0; the value is then fixed.
- **State:** the observed soma voltage at the probe time, split into 4 quantile bins.
- **Kernels:** reverse correlation, `K_s(τ|b) = mean_{t∈b} ξ_s(t) · y(t+1+τ) / ε`, for lags 0–148 ms.
- **Window:** for each pair of sites, the lags where both pooled kernels exceed 25% of their own peak.
- **Statistic:** `S` = mean over all 28 site pairs of the mean over bins and window lags of `Δ_b(τ)²`. Also reported for same-branch pairs (4) and cross-branch pairs (24) separately.

## Null

Permute the bin labels across probe times (200 permutations), recompute `S`. The invariant is **rejected** if `S_obs ≥ μ_null + 3σ_null`.

## Reported, not gated

- The same statistic on noise-free simulations of the same probed recordings (no OU, no measurement noise): what the noisy observer is trying to see.
- Per-bin kernel decay: the time for each pooled-normalised kernel to fall from its peak to 1/e, per bin, for T2 against T0.

## Fresh seeds

2000–2009 (10 recordings, each simulated as all four truths). 95% intervals are bootstrapped over recordings (2,000 resamples).

## Gates

- **Q0 — calibrated.** The invariant is rejected in ≤ 1 of 10 for T0 and ≤ 1 of 10 for T1.
- **Q1 — the probe sees the dendritic nonlinearity in ordinary activity.** The invariant is rejected in ≥ 8 of 10 for T2.
- **Reported, not a gate: is it the dendrite or any dynamics after a gain?** T1f against T2, as `z = (S − μ_null)/σ_null`, overall and for same-branch and cross-branch pairs. I could not state in advance a pattern that would separate them cleanly, so this is description, not a test. A rejection under Q1 means "not a pointwise gain at the readout", and T1f shows how much of that an output filter alone produces.

## Predictions, written before any code

Q0 pass, since the invariant is exact and the null matches the estimator. Q1 uncertain: soma voltage is a blunt label for branch state, and the probes are small. T1f rejects.
