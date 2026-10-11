# SihtiSort

**A spike sorter that remembers the previous spikes.**

A neuron's spike is not a fixed stamp. Martin-Burgos, Juavinett, Rivière, Hammonds & Voytek ([bioRxiv, Sept 2026](https://doi.org/10.64898/2026.09.15.751814)) found that 30 of 40 neurons switch between distinct spike shapes, often abruptly, and that one cell's own spikes can differ more than two different cells do. A sorter that judges each spike alone splits such a neuron into two "units".

SihtiSort tests whether remembering fixes that. If two clusters sit at the same place on the probe, never fire too close together, and take turns in long blocks instead of interleaving, then the hypothesis is *one neuron in two moods*, and it becomes one address with two states.

![one neuron through time](results/mood_timeline.png)

*One mood-switching neuron (post-hoc sparse recording, seed 11). Top: the snapshot sorter spreads it across units. Middle: SihtiSort holds 627 of its 633 spikes at one address and decodes its state. Bottom: the truth.*

## Result, in one paragraph

**The preregistered test failed.** On the five test recordings, SihtiSort recovered 7 of 20 mood neurons as one unit, against 6 of 20 for the snapshot-plus-merge baseline; the gate needed +25 points. The post-hoc analysis says why: neurons were placed at random, and 14 of the 20 mood neurons had another neuron within half a channel. No sorter recovered those 14, and every sorter recovered the other 6, so the test set held almost no case where the idea could matter. On recordings with neurons at least one channel apart (post hoc, not a gate result), SihtiSort recovered **20/20** against **14/20**, and **10/10** strong mood neurons against **4/10**. Removing the temporal rule drops that to 3/10. False merges did not rise (2 vs 3).

## How it works

The vocabulary maps onto the conversation it came from: addresses that open and close, guesses that get falsified, and a residue that gets purified.

1. **First look.** Detect spikes and cluster each one as a snapshot (PCA plus a Gaussian mixture, with the number of clusters chosen by BIC). This is the baseline sorter.
2. **Same-shape merge.** Clusters at the same place with correlated templates and no refractory clash are joined. This is the standard auto-merge in current sorters.
3. **Hypothesis: same neuron, different state.** For two clusters at the same place, compute the *switch ratio* R: how often the time-ordered union changes label, divided by how often it would if the two trains interleaved independently. A neuron changing mood gave R = 0.01–0.08 in these recordings. Two independent neurons give R ≈ 1. If R ≤ 0.3 and the refractory check passes, they become two states of one address.
4. **The sieve.** Every spike is explained by its best (address, state) template and subtracted, leaving a residue. Detection runs again on the residue to find spikes that were hidden under others.
5. **Look again.** Each address is re-split on *purified* snippets: its own spike plus the residue, with the neighbours' spikes removed. Every part goes back through steps 2–3 as a new hypothesis.
6. **Open new addresses** from whatever nothing explained, if it behaves like a neuron.
7. **Memory.** Each address carries a belief about its own current state, which fades toward uncertainty over about 5 s. The belief biases the next assignment, and the final state sequence comes from Viterbi decoding.

Every hypothesis tested in steps 2, 3, 5 and 6 is written to a **ledger** with its numbers and verdict: "could it be this? no, they interleave. Could it be that? Yes."

## Scoring correction (11 Oct, found by Sol)

The protocol says spikes match ground truth one-to-one, but the first evaluator matched each true spike to its nearest sorted spike, so one sorted spike could be counted twice where spikes overlap. [diagnostics/rescore.py](diagnostics/rescore.py) reruns the sorters (they are deterministic) and scores them both ways: the old scoring reproduces the original files exactly, and the one-to-one scoring changes no gate verdict. Overlap recall falls (SihtiSort 0.82 → 0.78 on the test set), the test-set baselines gain two false merges, and stable accuracy moves by under 0.01. All tables below use one-to-one scoring; the original files `results/test.json` and `results/sparse_posthoc.json` are kept as they were, and `results/*_rescored.json` hold both versions.

## Preregistered test (seeds 0–4, frozen commit `5c2153c`, run once)

The gates are in [PROTOCOL.md](PROTOCOL.md), committed before any code existed.

| Sorter | Mood neurons as one unit | Strong | Mild | Stable accuracy | False merges | Overlap recall |
|---|---|---|---|---|---|---|
| B1 snapshot | 8/20 | 3/10 | 5/10 | 0.738 | 23 | 0.13 |
| B2 snapshot + standard merge | 6/20 | 2/10 | 4/10 | 0.613 | 20 | 0.25 |
| **S SihtiSort** | **7/20** | 3/10 | 4/10 | **0.772** | **14** | **0.78** |
| S − temporal rule | 5/20 | 1/10 | 4/10 | 0.772 | 14 | 0.76 |
| S − memory | 7/20 | 3/10 | 4/10 | 0.772 | 14 | 0.78 |
| S − sieve | 6/20 | 2/10 | 4/10 | 0.613 | 20 | 0.25 |

| Gate | Verdict |
|---|---|
| G0 the problem exists | **Fail.** B1 recovers 40% of mood neurons, under the 50% bar, but its stable accuracy is 0.74, under the 0.8 bar: the baseline itself struggles on these recordings. |
| G1 main claim (+25 points over B2) | **Fail.** +5 points. False merges (14 vs 20) and stable accuracy (0.77 vs 0.61) do meet their conditions. |
| G2 the temporal rule does the work | Pass (S − temporal rule 25% vs B2 30%), but with G1 failed there is little to attribute. |
| G3 memory improves state decoding on mild neurons | **Not testable.** No sorter ever split a mild neuron into states, so there was nothing to decode. On strong neurons, state accuracy is 0.98–1.0 with or without memory. |
| G4 turn-taking pair | Merged once, at 0.86 channels apart (seed 3); never at ≥ 1.25. That is close to the stated prediction (merge at ≤ ~0.75). |
| G5 sieve sanity | Pass: overlap recall 0.78 vs 0.25. This is matching pursuit, which is prior art. |

## Post-hoc analysis (after the failure, labelled as such)

**Crowding** ([diagnostics/crowding.py](diagnostics/crowding.py)). With 10 neurons placed at random on 16 channels, 14 of 20 mood neurons had a neighbour within 0.5 channel. The protocol said neighbours would be close "sometimes", but in practice that was the norm.

| Test mood neurons | n | B1 | B2 | S | S − temporal rule |
|---|---|---|---|---|---|
| Crowded (neighbour < 0.5 ch) | 14 | 2 | 0 | 1 | 0 |
| Isolated (≥ 0.5 ch) | 6 | 6 | 6 | 6 | 5 |

**Oracle front end** ([diagnostics/oracle_front_end.py](diagnostics/oracle_front_end.py)). The merges were handed perfect clusters: every test-set mood neuron arrives split into its two true states. The temporal rule rejoined **14/20**, against 10/20 for the same-shape rule alone, and added **no** false joins (11 in both cases, all from the same-shape rule). Each of the 6 it missed has another neuron within 0.3 channel.

**Sparse recordings** (seeds 10–14, every neuron ≥ 1 channel from the others except the designed turn-taking pair; `python run_gates.py sparse_posthoc`):

| Sorter | Mood as one unit | Strong | Mild | Stable accuracy | False merges |
|---|---|---|---|---|---|
| B2 snapshot + standard merge | 14/20 | 4/10 | 10/10 | 0.887 | 3 |
| **S SihtiSort** | **20/20** | **10/10** | 10/10 | **0.986** | 2 |
| S − temporal rule | 13/20 | 3/10 | 10/10 | 0.986 | 2 |
| S − sieve | 19/20 | 9/10 | 10/10 | 0.887 | 3 |

Here the temporal rule is what recovers the strong mood neurons: 3/10 without it, 10/10 with it, and 9/10 even without the sieve. Mild moods were never split in the first place. The turn-taking pair (0.81–1.92 channels apart) was never merged.

## What the memory did

Almost nothing measurable. Each spike carries strong evidence about its own state, so the remembered state only decides near-ties. State accuracy on strong neurons in the sparse set is 0.994 with memory and 0.993 without. Memory should matter where single spikes are ambiguous: lower amplitudes, more similar moods, or real noise. These recordings don't contain that case.

## Deviations from the protocol

All were made on calibration seeds 100–102, before the frozen commit.

1. **Sieve acceptance.** The protocol's rule, "removes ≥ 50% of the snippet energy", rejected every spike that had another spike in its window. It was replaced by "removes ≥ 50% of the fitted template's own energy". Before the change, about 10% of every neuron's spikes went unexplained.
2. **"Look again" step added** (step 5 above). Without it, mood states arrived mixed with neighbouring neurons and the temporal test saw interleaving.
3. Residue clustering was allowed 1–10 clusters; the protocol did not fix this.
4. A peak-channel local-clustering front end was tried and removed, because it produced more false merges.

## What this is not

- **Synthetic.** The mood model is one reading of one preprint: two states, exponential dwell times around 8 s, firing rate independent of mood.
- **Not new machinery, except one test.** Residue subtraction is matching pursuit, as in current template-matching sorters. Refractory and cross-correlogram merges and drift correction exist in current sorters too. I have not checked whether any sorter already models within-unit waveform states over time. The switch-ratio test and the state-aware addresses are the parts specific to this repository.
- **Identifiability limit.** From spikes alone, "one neuron with two moods" and "two neurons at the same place taking turns" can look the same. The footprint is the only separator, and it fails below roughly one channel of separation.

## Commit, name the refuter, revise ([PROTOCOL_2.md](PROTOCOL_2.md))

Antti's bike ride: two coloured shapes far ahead became "two cyclists" at once, then "two signs" as he rode closer. The brain didn't say "I can't tell". It committed and stayed open to correction. So every "one neuron, two states" merge became a **commitment** carrying three things: a confidence (a runs test on the state sequence), a named **refuter** (one neuron keeps its firing rate across waveform states, per the preprint's η² = 0.013, so a ≥ 1.5× rate gap at z ≥ 3 means two neurons), and a status of *survived*, *refuted* (the address is split) or *untested from here* (with the extra recording time needed).

One design change was made on calibration seeds 40–41, before the frozen commit `af952bc`. The close turn-taking pair was never a two-state hypothesis, because the same-shape rule swallowed it. In commit mode, same-shape parts that take turns are now handed to the state rule. Protocol 2's thresholds are unchanged.

**Identifiability set** (seeds 20–31: neurons ≥ 1 channel apart, except the turn-taking pair, placed 0–0.5 channels apart):

| Gate | Result |
|---|---|
| R1 catches the false merge (pairs with true rate gap ≥ 1.5×) | **Fail.** Two merged pairs had a rate gap that large (2.0× and 2.7×), and neither ever became a two-state hypothesis. The first clustering lumped each pair together, and the look-again step didn't split it. A refuter only works on a hypothesis that exists. |
| R2 doesn't destroy true merges | **Pass.** 0 of 30 true mood merges wrongly split. |
| R3 "survived" means something (≥ 90% correct) | **Fail.** 29 of 34 correct (85%). All 5 wrong ones are turn-taking pairs whose rates differ by ≤ 1.37×. The rate refuter cannot see them, so "survived" overstated what was tested. |
| R4 the bike ride | **No revision ever happened.** The refuter never fired. Commitments reached "survived" after 20–55 s of recording. |

On the earlier sets nothing regressed: mood neurons kept as one unit stayed at 7/20 (test) and 20/20 (sparse), and sparse false merges went from 2 to 1.

**The lesson, which is the bike ride's too:** surviving a test that can't see the difference is not evidence. From half a mile away, "two cyclists" survives every look, because distance can't separate a cyclist from a sign. The refuter has to be one that can actually tell the alternatives apart.

**Post hoc: a position refuter** ([diagnostics/position_refuter.py](diagnostics/position_refuter.py), threshold not fixed in advance, n small). One neuron in two states sits at one place; two neurons sit at two. Each state's position was estimated as the amplitude-weighted channel centroid of its median spike:

| Commitments on the identifiability set | Gap between the two states' positions (channels) |
|---|---|
| 30 true one-neuron merges | median 0.024, max 0.097 |
| 6 true two-neuron merges (true offset → measured gap) | 0.32 → 0.30, 0.25 → 0.27, 0.23 → 0.27, 0.16 → 0.10, 0.12 → 0.09, 0.00 → 0.01 |

The three pairs ≥ 0.23 channels apart sit well clear of every one-neuron merge. The pairs at 0.12–0.16 channels overlap the one-neuron spread, and the pair at 0.00 is truly identical from here. So "getting closer" means sub-channel position, and below about a tenth of a channel this probe cannot tell; a denser probe could.

## The test that matters next

The preprint's **spe-1** dataset ([CRCNS](https://crcns.org/), analysis code at [voytekresearch/spikewise](https://github.com/voytekresearch/spikewise)) has simultaneous intracellular ground truth beside the Neuropixels recording. For the cells whose waveform switches state, the question is whether a sorter that remembers keeps each one as one unit where snapshot sorting splits it. Download needs a CRCNS account, and this repository does not include a loader yet.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q tests
python run_gates.py calibration      # seeds 100-102 (~1.5 min)
python run_gates.py test             # seeds 0-4, the preregistered test (~2.5 min)
python run_gates.py sparse_posthoc   # seeds 10-14, post hoc (~2.5 min)
python diagnostics/oracle_front_end.py
python diagnostics/crowding.py
python diagnostics/figure_timeline.py
python diagnostics/rescore.py         # one-to-one rescoring (~5 min)
python diagnostics/commit_and_revise.py    # PROTOCOL_2 (~9 min)
python diagnostics/position_refuter.py     # post hoc (~3.5 min)
```

Files: [sihtisort/synth.py](sihtisort/synth.py) (recording and ground truth) · [sihtisort/sort.py](sihtisort/sort.py) (B1, B2, SihtiSort) · [sihtisort/evaluate.py](sihtisort/evaluate.py) (scoring) · [run_gates.py](run_gates.py) · `results/*.json` (every number above, with the ledger of every hypothesis tested).
