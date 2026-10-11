"""Figure: one strong mood neuron through time, as the snapshot sorter and SihtiSort see it.

Uses the POST HOC sparse recordings (neurons >= 1 channel apart).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort import sort as S                      # noqa: E402
from sihtisort.evaluate import match_idx             # noqa: E402
from sihtisort.synth import FS, make_recording   # noqa: E402

import json   # noqa: E402
# first strong mood neuron that SihtiSort holds as one address with two states
res = json.load(open('results/sparse_posthoc.json'))['per_seed']
seed, nid = next((int(s), n['id']) for s, r in res.items() for n in r['S']['neurons']
                 if n['kind'] == 'mood_strong' and 'state_acc' in n)
data, gt, meta = make_recording(seed, min_sep=1.0)
base = S.snapshot_sort(data, 0)
s_out = S.run_s(data, base=base)
g = gt['unit'] == nid
gt_t, gt_s = gt['t'][g], gt['state'][g]

fig, ax = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
# snapshot: which cluster each spike landed in
j = match_idx(gt_t, base['t'])
ok = j >= 0
labs = base['label'][j[ok]]
uniq = [u for u, c in zip(*np.unique(labs, return_counts=True)) if c > 0.05 * ok.sum()]
for k, u in enumerate(uniq):
    m = labs == u
    ax[0].plot(gt_t[ok][m] / FS, np.full(m.sum(), k), '|', ms=10, color=f'C{k}')
ax[0].set_yticks(range(len(uniq))); ax[0].set_yticklabels([f'unit {u}' for u in uniq])
ax[0].set_title('Snapshot sorter: one neuron is split across units', loc='left', fontsize=10)
# SihtiSort: one address, its decoded state
order = np.argsort(s_out['t'])
st, su, ss = s_out['t'][order], s_out['unit'][order], s_out['state'][order]
j = match_idx(gt_t, st); ok = j >= 0
addr = np.bincount(su[j[ok]]).argmax()
mm = (su[j[ok]] == addr)
dec = ss[j[ok]][mm]
ax[1].plot(gt_t[ok][mm] / FS, dec, '|', ms=10, color='C2')
ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(['state a', 'state b'])
ax[1].set_title(f'SihtiSort: one address ({mm.sum()} of {len(gt_t)} spikes), its remembered state (names a/b are arbitrary)', loc='left', fontsize=10)
ax[2].plot(gt_t / FS, gt_s, '|', ms=10, color='0.3')
ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(['true a', 'true b'])
ax[2].set_title('Ground truth: the neuron\'s mood', loc='left', fontsize=10)
ax[2].set_xlabel('time (s)')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
Path('results').mkdir(exist_ok=True)
fig.savefig('results/mood_timeline.png', dpi=130)
print('neuron', nid, 'address', addr, 'units in snapshot', uniq)
