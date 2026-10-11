"""POST HOC diagnostic, written after the test run failed G1 (see README).

Question: is the temporal-context rule itself sound, so that the failure lies in the
snapshot front end? Hand the rule a perfect snapshot sorter: every detected event is
labelled with its true neuron and, for mood neurons, its true state, so each mood neuron
arrives split into two clean state clusters. Then run the same merges as SihtiSort
(standard merge, then temporal-context merge) on the real snippets and count:
  - mood neurons whose two states are rejoined into one address
  - addresses that join two different neurons (the turn-taking pair counted apart)
The same is done with the temporal rule switched off, as the control.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort import sort as S                       # noqa: E402
from sihtisort.evaluate import match_idx               # noqa: E402
from sihtisort.synth import make_recording             # noqa: E402


def oracle_labels(t, gt):
    j = match_idx(t, gt['t'])                         # nearest true spike for each detection
    keep = j >= 0
    lab = gt['unit'][j[keep]] * 2 + gt['state'][j[keep]]
    return keep, lab


def run(seed):
    data, gt, meta = make_recording(seed)
    sigma = S.noise_sigma(data)
    t = S.detect(data, sigma)
    snips = S.snippets(data, t)
    keep, lab = oracle_labels(t, gt)
    t, snips = t[keep], snips[keep]
    kinds = {nu['id']: nu['kind'] for nu in meta['neurons']}
    out = {}
    for block in (True, False):
        ledger = []
        lab2, addresses = S.group(t, lab, snips, ledger, block)
        # membership: which true (neuron, state) clusters ended up in each address
        member_sets = []
        if block:
            unit_to_addr = {}
            for a, ad in enumerate(addresses):
                for m in ad['members']:
                    unit_to_addr[m] = a
            addr = np.array([unit_to_addr[int(l)] for l in lab2])
        else:
            addr = lab2
        for a in np.unique(addr):
            src = lab[addr == a]
            n_by = {int(c): int((src == c).sum()) for c in np.unique(src)}
            member_sets.append(n_by)
        rejoined, false_join, turn_join = 0, 0, False
        for nid, kind in kinds.items():
            if kind.startswith('mood'):
                both = [ms for ms in member_sets
                        if ms.get(2 * nid, 0) > 0.5 * (lab == 2 * nid).sum()
                        and ms.get(2 * nid + 1, 0) > 0.5 * (lab == 2 * nid + 1).sum()]
                rejoined += bool(both)
        for ms in member_sets:
            tot = sum(ms.values())
            neurons = {c // 2 for c, n in ms.items() if n >= 0.2 * tot}
            if len(neurons) >= 2:
                if sorted(kinds[n] for n in neurons) == ['turn_a', 'turn_b']:
                    turn_join = True
                else:
                    false_join += 1
        out['temporal' if block else 'standard only'] = dict(
            mood_rejoined=rejoined, false_joins=false_join, turn_pair_joined=turn_join,
            temporal_tests=[r for r in ledger if r['test'] == 'same neuron, different state'
                            and r.get('verdict') != 'reject: different place'])
    return dict(turn_offset=meta['turn_offset'], **out)


if __name__ == '__main__':
    res = {s: run(s) for s in [0, 1, 2, 3, 4]}
    tot = {k: dict(mood_rejoined=sum(r[k]['mood_rejoined'] for r in res.values()),
                   false_joins=sum(r[k]['false_joins'] for r in res.values()),
                   turn_pair_joined=[(round(r['turn_offset'], 2), r[k]['turn_pair_joined']) for r in res.values()])
           for k in ('temporal', 'standard only')}
    print(json.dumps(tot, indent=1))
    Path('results').mkdir(exist_ok=True)
    with open('results/oracle_front_end.json', 'w') as f:
        json.dump(dict(note='POST HOC, after the test run', totals=tot,
                       per_seed={str(k): v for k, v in res.items()}), f, indent=1,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
