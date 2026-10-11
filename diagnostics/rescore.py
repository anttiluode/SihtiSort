"""Rescore every set with one-to-one spike matching (the protocol's rule).

Sol found that evaluate.py matched each true spike to its nearest sorted spike without
stopping one sorted spike from being counted twice. Sorters are rerun (they are
deterministic) and scored both ways; the original results files are left untouched.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_gates as RG                                  # noqa: E402
from sihtisort import sort as S                         # noqa: E402
from sihtisort.evaluate import score                    # noqa: E402
from sihtisort.synth import make_recording              # noqa: E402
import numpy as np                                      # noqa: E402


def outputs(seed, min_sep):
    data, gt, meta = make_recording(seed, min_sep=min_sep)
    base = S.snapshot_sort(data, 0)
    outs = {'B1': dict(t=base['t'], unit=base['label'], state=np.zeros(len(base['t']), int)),
            'B2': S.run_b2(data, base=base)[0], 'S': S.run_s(data, base=base),
            'S-block': S.run_s(data, base=base, block=False), 'S-memory': S.run_s(data, base=base, memory=False),
            'S-sieve': S.run_s(data, base=base, use_sieve=False)}
    return outs, gt, meta


for which in ['test', 'sparse_posthoc']:
    new, old = {}, {}
    for seed in RG.SETS[which]:
        outs, gt, meta = outputs(seed, RG.MIN_SEP.get(which))
        new[seed] = {k: score(o, gt, meta, one_to_one=True) for k, o in outs.items()}
        old[seed] = {k: score(o, gt, meta, one_to_one=False) for k, o in outs.items()}
    Pn = {v: RG.pooled(new, v) for v in RG.VARIANTS}
    Po = {v: RG.pooled(old, v) for v in RG.VARIANTS}
    out = dict(note='rescored with one-to-one matching after Sol found the double-count bug',
               pooled_one_to_one=Pn, gates_one_to_one=RG.gates(Pn), pooled_old_scoring=Po)
    json.dump(out, open(f'results/{which}_rescored.json', 'w'), indent=1,
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(which)
    for v in RG.VARIANTS:
        a, b = Po[v], Pn[v]
        print(f"  {v:9s} mood {a['mood_one_unit']:2d}->{b['mood_one_unit']:2d}  strong {a['strong_one_unit']}->{b['strong_one_unit']}"
              f"  stable {a['stable_mean_acc']:.3f}->{b['stable_mean_acc']:.3f}  FM {a['false_merges']}->{b['false_merges']}"
              f"  overlap {a['overlap_recall']:.3f}->{b['overlap_recall']:.3f}")
    for k, g in RG.gates(Pn).items():
        print('  ', k, g.get('passed', g.get('merge_rule_failed')))
