# SihtiSort protocol — fixed before any test result

Written 11 October 2026, before the sorter or the benchmark existed. The commit that adds this file precedes every result file in the repository. Thresholds below are set a priori, not tuned. The code is debugged on calibration seeds 100–102 only; test seeds 0–4 are run once, after the code is frozen, and the frozen commit is recorded in the results.

## The question

Martin-Burgos, Juavinett, Rivière, Hammonds & Voytek ([bioRxiv 2026.09.15.751814](https://doi.org/10.64898/2026.09.15.751814)) report that one neuron's action-potential waveform switches between discrete states (30 of 40 neurons multimodal; switches often abrupt), and that within-cell differences can exceed between-cell differences. Their discussion predicts that spike sorting, which judges each spike as a snapshot, will split such a neuron into two units.

SihtiSort tests one idea against that: **a sorter that remembers the previous spikes can recognise the same sender through a change of state.** Two clusters at the same place on the probe that take turns in long blocks are tested as one neuron in two states, not two neurons.

## Synthetic recording (ground truth)

- 16-channel linear probe, 20 kHz, 60 s; noise Gaussian, smoothed across channels and lightly in time, unit standard deviation per channel.
- 10 neurons per recording:
  - **4 mood neurons** with two waveform states and abrupt switches (two-state process, exponential dwell, mean 8 s, minimum 1 s). Two are *strong* (state B: amplitude ×0.6, trough width ×1.6, repolarization ×1.3) and two *mild* (×0.8, ×1.3, ×1.1). The spatial footprint width also changes by a random ±15% between states, so the footprint is not identical across states.
  - **2 turn-taking neurons**: distinct neurons a random 0.5–2.0 channels apart that fire only in alternating blocks (same block process). This is the designed hard case: from spikes alone it can resemble one neuron with two moods.
  - **4 stable neurons**.
- Firing: Poisson with 2 ms dead time, 3–12 Hz, rate independent of mood state. Per-spike amplitude jitter 7%.
- Positions are random along the probe, so unrelated neurons sometimes sit close together.

## Sorters

- **B1 snapshot**: threshold detection (4.5 × MAD per channel, 0.5 ms dead time), 48-sample × 16-channel snippets, PCA to 10 dimensions, Gaussian mixture with the number of clusters chosen by BIC over 4–20.
- **B2 snapshot + standard merge**: B1 then merge pairs whose footprints agree (cosine ≥ 0.95), whose templates correlate (≥ 0.9), and whose union passes the refractory check. This stands in for the auto-merge step in current sorters.
- **S SihtiSort**: B2's units, then:
  1. **Temporal-context merge.** For two units at the same place (footprint cosine ≥ 0.95) with ≥ 30 spikes each, a passing refractory check, and a *switch ratio* R ≤ 0.3, the hypothesis "one neuron in two states" is accepted and the two units become two states of one address. R is the number of label switches in their time-ordered union divided by the number expected if the two trains interleaved independently, 2pq(N−1). Every tested hypothesis, accepted or rejected, is written to a ledger.
  2. **The sieve.** Matching pursuit on the residue: each event is explained by the best (address, state) template with a fitted amplitude in [0.7, 1.3] and a ±2-sample shift, accepted if it removes ≥ 50% of the snippet energy, then subtracted. Detection is repeated on the residue for 3 passes. Events nothing explains are clustered again and may open new addresses (≥ 30 spikes, refractory check), then the sieve is rerun.
  3. **Memory.** While sieving in time order, each address keeps a belief over its own states, carried forward with a transition that relaxes toward uniform with time constant 5 s since its last spike. The log-belief is added to each (address, state) score. Final state labels per address come from Viterbi decoding with the same transition.
- Refractory check (all sorters): fraction of ISIs < 1.5 ms in the union ≤ max(of the two parts) + 0.005 and ≤ 0.01.

Ablations: **S−block** (no temporal-context merge), **S−memory** (uniform state prior, per-spike state argmax), **S−sieve** (merges and memory, but spikes and snippets from the original detection only).

## Scoring

- Spikes match ground truth within ±0.4 ms, one-to-one.
- Accuracy of unit u for neuron g: matches / (n_g + n_u − matches). Each neuron's best unit defines its score.
- A neuron is **recovered as one unit** if its best accuracy ≥ 0.8.
- A **false merge** is a unit in which two different neurons each supply ≥ 20% of the unit's matched spikes. The designed turn-taking pair is counted separately.
- **State accuracy**: for mood neurons recovered as one unit whose unit has ≥ 2 states, agreement between decoded and true state on matched spikes, each decoded state mapped to its majority true state.
- **Overlap recall**: recall of ground-truth spikes that lie within 1 ms of another neuron's spike.

## Gates (pooled over test seeds 0–4: 20 mood neurons, 20 stable, 5 turn-taking pairs)

- **G0 — the problem exists.** B1 recovers ≤ 50% of mood neurons as one unit, while its mean best accuracy on stable neurons is ≥ 0.8. If B1 already recovers the mood neurons, there is nothing to fix and that is the result.
- **G1 — the main claim.** S recovers mood neurons as one unit at least 25 percentage points more often than B2; S has at most 2 more false merges than B2 in total (turn-taking pair excluded); S's mean stable accuracy is no more than 0.02 below B2's.
- **G2 — what does the work.** S−block recovers mood neurons at most 10 points more often than B2. If the sieve alone recovers them, the temporal-context merge is not what matters.
- **G3 — memory.** On mild mood neurons recovered as one unit, S's state accuracy exceeds S−memory's by ≥ 5 points. Strong mood neurons are reported but not gated.
- **G4 — identifiability limit (reported, prediction stated).** Prediction: S merges the turn-taking pair when it sits ≤ ~0.75 channel apart and not at ≥ 1.25 channels. If S merges pairs ≥ 1.25 channels apart in ≥ 2 of 5 recordings, the footprint gate is too loose and that counts as a failure of the merge rule.
- **G5 — sieve sanity (prior art, not a claim).** S's overlap recall ≥ S−sieve's. Residue subtraction is matching pursuit, used by existing template-matching sorters.

## What would not be a claim even if every gate passes

The recording is synthetic and its mood model follows one reading of one preprint. The merge rule assumes a neuron's position on the probe changes less between states than between neighbouring neurons. Real data — the preprint's spe-1 set, with intracellular ground truth — is the test that matters; this repository only shows whether the idea works where it should.
