# Protocol 3 — two channels: a committed address and a remainder

Written 11 October 2026, after Protocol 2 and before any code for this addendum. Thresholds are set here. Code may be debugged on seeds 70–71 only; seeds 50–61 are run once on frozen code.

## Why

Protocol 2 failed for two reasons, both pointed out by Sol:

- **The alternative was erased upstream.** Two turn-taking neurons were lumped into one unit before any refuter could look, so "two neurons" was never a hypothesis.
- **The refuter could not see the difference.** Rate is blind when the two neurons fire at similar rates.

The J-space paper (Gurnee et al. 2026, §4.1.1, A.15) shows what a language model does with an ambiguous input. Its workspace commits to one reading while the rest of the activation still carries the unresolved mixture, for a while. Two channels: one to act on, one that still remembers the other possibility.

## The two channels

1. **Committed address.** SihtiSort in Protocol 2 mode. This is what the sorter acts on and reports.
2. **Remainder.** Every committed address keeps a split of itself alive:
   - for a two-state address, its two states;
   - for a one-state address with ≥ 60 spikes, a forced two-way split (Gaussian mixture, k = 2) of its *purified* snippets (residue plus its own fitted template).

The remainder is tested with two refuters of "one neuron":

- **Position** (from Protocol 2's post-hoc finding): each spike's channel centroid, taken from its purified snippet. The two parts disagree if their mean positions differ by ≥ **0.15 channels** with z ≥ **3**. The value 0.15 sits above the largest one-neuron gap seen post hoc (0.097).
- **Rate** (Protocol 2's refuter): applied only when the parts take turns (switch ratio ≤ 0.3).

If either refuter fires, the address is **reopened**: its two parts become two addresses. Otherwise the remainder's status is *agrees* (position test had power: SE ≤ 0.05 channels) or *silent* (could not test from here).

## Online, not retrospective

Sol noted that Protocol 2's evidence-over-time was reconstructed after the fact. Here the whole pipeline is rerun on the first 10, 20, …, 60 s of each recording, using no data beyond the prefix. Each prefix produces its own commitments, remainder tests and reopenings: what the sorter believed at that moment.

## Fresh identifiability set: seeds 50–61

Same generator as seeds 20–31: neurons ≥ 1 channel apart, except the turn-taking pair at 0–0.5 channels apart, rates 3–12 Hz each.

## Gates (full 60 s recordings unless stated)

- **P1 — reopens false merges that can be seen.** Among turn-taking pairs that Protocol 2 mode leaves merged *and* that are separable in principle (true offset ≥ 0.25 channels or true rate ratio ≥ 1.5), the two-channel sorter ends with them apart in ≥ 70%.
- **P2 — keeps true merges.** Mood neurons recovered as one unit drop by at most 1 against Protocol 2 mode, pooled.
- **P3 — doesn't shatter single neurons.** Stable neurons recovered as one unit drop by at most 2 against Protocol 2 mode, pooled (48 stable neurons).
- **P4 — revision happens online (reported, with one condition).** Count turn-taking pairs that are merged at some prefix and separated at the end. Condition: no turn-taking pair is separated at an earlier prefix and wrongly re-merged at 60 s.
- **P5 — honest about what it cannot see (reported).** For pairs that stay merged, the remainder's status (*silent* or *agrees*) against their true offset and rate ratio.

Regression check on the earlier test (0–4) and sparse (10–14) sets: mood and stable neurons recovered as one unit must not drop by more than 1 per set.

## Not a claim

Synthetic data; the position refuter assumes a neuron's footprint centroid is stable across its states, which electrode drift would violate in real recordings.
