# Protocol R1 — what the best matrix neuron cannot explain

Written 12 October 2026, before any code for this line. Thresholds are set here. Code may be debugged and tuned on calibration seeds 900–902 only; the fresh seeds below are run once on a frozen commit.

This is a separate line from Protocols 1–7 (spike sorting). It asks the question behind the whole repository's name: the sieve keeps the residue, but a residue only means something relative to the model whose error it is.

## The question

A passive dendrite is exactly a matrix: `v[t+1] = P v[t] + X u[t]`, a heat flow on a weighted graph, the same object as Sihti's image diffusion. That part is textbook. The bet is that a real neuron is *not* that, and specifically that its nonlinearity sits **inside** the operator (`P` depends on the state, as with NMDA receptors) rather than **between** fixed linear operators (a fixed filter, then a static nonlinearity, then a fixed filter — the structure of LN cascades, Poirazi-style two-layer neurons and scattering transforms).

So: fit the best matrix neuron, and the best fixed-filter cascade, to a neuron observed only at the soma. Take what each fails to explain. Ask

1. whether that residue has structure the model's own calibrated null cannot produce, and
2. whether that structure predicts the neuron's response to an intervention it was never trained on.

## Truths (same seed ⇒ same morphology, synapses and input trains; only the nonlinearity differs)

A soma plus 4 branches (8–16 compartments each, random axial conductance per branch), 2 input sites per branch on the distal half, subthreshold (no spike generation). Current-based AMPA inputs at every site. Intrinsic noise: an Ornstein–Uhlenbeck current in every compartment; measurement noise at the soma.

| Truth | What it is | Class |
|---|---|---|
| T0 passive | the cable alone | exactly a matrix neuron |
| T1 passive + static output nonlinearity | the soma voltage is read through a saturating `tanh` | fixed filter → static nonlinearity: in the cascade class |
| T2 NMDA | each site also has an NMDA conductance with Mg²⁺ block | nonlinearity inside the operator |

The observer knows the presynaptic spike times at each site and which branch each site is on (an uncaging or optogenetic experiment), and sees only the soma voltage, sampled at 1 kHz.

## Regimes

- **Regime A (training and held-out):** independent Poisson input at every site. 0.5–40 s is training, 40.5–50 s is held-out.
- **Regime B (novel intervention, 50.5–60 s):** branch bursts — every few hundred ms one branch gets a synchronous burst on both its sites — over background Poisson at half rate. Nothing like this pattern is deliberately present in A.

## Observer models (fitted on regime-A training data only)

- **M_lin, the matrix neuron.** Each site's spike train is filtered by a bank of causal exponentials with log-spaced time constants; ridge regression to the soma voltage. A passive cable's response is a sum of exponentials, so this class contains T0 up to the basis approximation.
- **M_LN, the best fixed-filter cascade.** The same exponential bank → a hidden layer of `tanh` units with trained weights → each unit's output filtered by a second fixed exponential bank (including an instantaneous term) → linear readout, plus the linear passthrough of M_lin, plus a cubic output polynomial. Fitted by L-BFGS from a fixed initialisation. This contains T1 and Poirazi-style branch-wise LN structure. Every filter in it is fixed in time; the nonlinearity only sits between them.

## Residue probes (trained on training residue, scored on held-out regime-A residue)

- **P_gen (generic):** a fixed set of random `tanh` features of the exponential bank, ridge to the residue.
- **P_state (the bet's own form):** each branch's input features multiplied by that branch's estimated state (its linear drive from M_lin), ridge to the residue. To first order this is an operator whose coefficients on each time constant move with the branch's state.

Structure score `S` = held-out R² of the probe on the held-out residue.

## The calibrated null (the Protocol 7 lesson)

For each recording and each model: generate synthetic soma data as the fitted model's own prediction plus noise with the **same power spectrum** as the real residue (phase-randomised, separately for training and held-out); **refit the model with the identical procedure**, recompute residues, retrain the probe, and score it. 20 draws. The residue is called **structured** if `S_obs ≥ max(0.01, μ_null + 3σ_null)`.

Reported alongside, not gated: a naive null that skips the refit (the probe scored on phase-randomised residue against the real fit).

## Fresh seeds

Seeds 1000–1009 (10 recordings, each simulated as all three truths). The recording is the resampling unit; 95% intervals are bootstrapped over recordings (2,000 resamples).

## Gates

- **R0 — the instrument is calibrated.** T0 with M_lin: P_gen structured in ≤ 1 of 10. T1 with M_LN: P_gen structured in ≤ 1 of 10. If R0 fails, R2 cannot be interpreted.
- **R1 — the instrument sees nonlinearity.** P_gen structured with M_lin in ≥ 8 of 10 for T1, and in ≥ 8 of 10 for T2.
- **R2 — the bet: the best fixed-filter cascade leaves structure on the NMDA neuron.** T2 with M_LN: P_gen structured in ≥ 8 of 10.
- **R3 — the structure predicts a new intervention.** On regime B, `Δ = R²_B(M_LN + P_state) − R²_B(M_LN)` (P_state trained on regime-A residue only). Pass if on T2 the mean Δ > 0 with a 95% interval excluding 0, **and** the paired difference Δ(T2) − Δ(T1) has a 95% interval excluding 0 (the probe helps where there is a state-dependent operator, not merely anywhere).
- **Reported:** P_state structure for every truth/model, the naive null's verdicts, R²_A and R²_B for all models with and without each probe.

## Predictions, written before any code

R0: likely pass, but the exponential basis only approximates the cable's modes, and that approximation error is structured residue the null cannot contain — the same failure as Protocol 7's drift estimate. R1: pass. R2: likely pass; Beniaguev, Segev & London (2021) needed several layers for an NMDA neuron and one for a passive one, but this toy is smaller than theirs. R3: genuinely open; a residue model fitted under one input regime may describe that regime rather than the mechanism, and regime B is out of distribution.
