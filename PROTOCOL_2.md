# Protocol 2 — commit, name the refuter, revise

Written 11 October 2026, after the first test (PROTOCOL.md) and before any code for this addendum. Thresholds are set here, a priori.

## The idea

Antti, biking on Ring Road III: two coloured shapes half a mile ahead became "two cyclists" at once, and became "two blue-and-yellow signs" as he rode closer. The brain did not answer "I can't tell". It committed to the most likely story and kept it open to correction.

SihtiSort v0 commits silently. Every "one neuron, two states" merge it makes is final. This addendum makes each merge a **commitment** that carries three things:

1. **Confidence**: how strongly the two states take turns. This is the Wald–Wolfowitz runs test on the time-ordered state labels; z is the number of runs below what independent interleaving predicts.
2. **A named refuter**: an observation that one neuron cannot produce but two neurons usually would.
3. **A status**: *survived* (the refuter had enough data to fire and did not), *refuted* (it fired, so the address is split back into two neurons), or *untested from here* (not enough data yet, together with an estimate of how much more recording would decide it).

## The refuter

The preprint behind this repository (Martin-Burgos et al. 2026) found that a spike's waveform state did not predict its inter-spike interval (median η² = 0.013): **a neuron keeps its firing rate across waveform states.** Two different neurons taking turns usually fire at different rates.

For an address with two states, the decoded state sequence is cut into runs. Each state's rate is Σ(spikes in run − 1) / Σ(run duration) over its runs of ≥ 2 spikes. With k_a and k_b the summed interval counts:

- log ratio L = |log(r_a / r_b)|, standard error SE = √(1/k_a + 1/k_b), z = L / SE
- **refuted** if z ≥ 3 **and** r_a / r_b (or its inverse) ≥ 1.5. The ratio floor is there because a low-amplitude state loses some spikes below threshold and so looks slower; up to about a third lost is tolerated.
- **survived** if not refuted and SE ≤ 0.135, i.e. the data could have detected a 1.5× rate difference at z = 3.
- **untested from here** otherwise, reported with the number of extra spikes per state needed to reach SE ≤ 0.135.

A refuted address is split: each state becomes its own address. Addresses with more than two states test every state against the largest one.

## Evidence over time

The same statistics are recomputed on growing prefixes of the recording (every 5 s). This is the bike ride: commitment, confidence, and the moment of revision, if one comes.

## Identifiability set (new seeds 20–31)

Same generator as the sparse post-hoc set: every neuron ≥ 1 channel from the others. The one change is that the turn-taking pair is placed 0–0.5 channels apart, so that it sits where SihtiSort v0 would merge it. Rates are drawn as before (3–12 Hz each, independently), so some pairs fire at similar rates. In those pairs the refuter is blind, and saying so is the correct behaviour.

## Gates

On the identifiability set:

- **R1 — catches the false merge.** Among turn-taking pairs that v0 merges *and* whose true rates differ by ≥ 1.5×, revision splits ≥ 70%.
- **R2 — does not destroy true merges.** Among mood neurons that v0 holds as one multi-state address, revision wrongly splits ≤ 10%.
- **R3 — labels mean something.** Merges labelled *survived* are correct (both states from one true neuron) at ≥ 90%. Merges labelled *untested from here* are reported with their correctness; no threshold, because the claim is only that they were flagged.
- **R4 — the bike ride (reported).** On the evidence-over-time record: how many commitments are revised, and how many seconds of recording each needed.

On the earlier test and sparse sets (reported, post hoc): mood neurons recovered as one unit must not drop by more than one per set after revision.

## What would not be a claim

The refuter is only as good as its assumption. If real neurons change rate with waveform state, this refuter fires on true merges. That is why the preprint's η² finding is cited above, and why spe-1 with intracellular ground truth remains the test that matters.
