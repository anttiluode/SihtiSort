"""POST HOC (after PROTOCOL_2 failed R1, R3, R4): would a position refuter see what the rate refuter missed?

One neuron in two waveform states sits at one place; two neurons taking turns sit at two places,
here 0-0.5 channels apart. For every commitment SihtiSort makes on the identifiability set, estimate
each state's position on the probe (amplitude-weighted channel centroid of its median spike) and
compare the gap for true one-neuron merges against true two-neuron merges.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort import sort as S                          # noqa: E402
from sihtisort.evaluate import global_match              # noqa: E402
from sihtisort.synth import make_recording               # noqa: E402


def centroid(T):
    f = T.max(0) - T.min(0)
    f = np.clip(f - 0.2 * f.max(), 0, None)      # ignore the far tails
    return float((f * np.arange(len(f))).sum() / f.sum())


rows = []
for seed in range(20, 32):
    data, gt, meta = make_recording(seed, min_sep=1.0, turn_range=(0.0, 0.5))
    base = S.snapshot_sort(data, 0)
    out = S.run_s(data, base=base, commit=True)
    o = np.argsort(out['t'], kind='stable'); t, u, s = out['t'][o], out['unit'][o], out['state'][o]
    gm = global_match(gt['t'], t)
    inv = np.full(len(t), -1); ok = gm >= 0; inv[gm[ok]] = gt['unit'][ok]
    kinds = {n['id']: n['kind'] for n in meta['neurons']}
    for c in out['commitments']:
        a, b = c['states']
        ma = (u == c['address']) & (s == a); mb = (u == c['address']) & (s == b)
        ga = np.bincount(inv[ma][inv[ma] >= 0]).argmax(); gb = np.bincount(inv[mb][inv[mb] >= 0]).argmax()
        Ta = np.median(S.snippets(data, t[ma]), 0); Tb = np.median(S.snippets(data, t[mb]), 0)
        rows.append(dict(seed=seed, truth='one neuron' if ga == gb else 'two neurons', kind=kinds[int(ga)],
                         status=c['status'], position_gap=round(abs(centroid(Ta) - centroid(Tb)), 3),
                         true_offset=round(meta['turn_offset'], 3) if ga != gb else 0.0))
    print('seed', seed, 'done', flush=True)

one = [r['position_gap'] for r in rows if r['truth'] == 'one neuron']
two = [(r['position_gap'], r['true_offset']) for r in rows if r['truth'] == 'two neurons']
summary = dict(one_neuron_gap_max=max(one), one_neuron_gap_median=float(np.median(one)), n_one=len(one),
               two_neuron_gaps=sorted(two), n_two=len(two))
print(json.dumps(summary, indent=1))
json.dump(dict(note='POST HOC', summary=summary, rows=rows), open('results/position_refuter.json', 'w'), indent=1,
          default=lambda o: o.item() if hasattr(o, 'item') else str(o))
