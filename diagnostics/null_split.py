"""Two cheap diagnostics Sol asked for (after Protocol 6), no sorting involved.

A. Interpolation and centroid bias: noise-free templates at known positions, shifted by known drift,
   then (i) centroid minus drift, (ii) registered back by interpolation. How much error remains,
   as a function of where on the channel grid the neuron sits?

B. The forced split under the one-neuron null: one neuron, many noisy spikes, the exact remainder
   procedure (split fitted on even spikes, test on odd spikes, template gap + bootstrap SE).
   How often does it declare two neurons when there is one?
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort import sort as S                       # noqa: E402
from sihtisort.synth import FS, N_CH, template         # noqa: E402

out = {}

# ---------------------------------------------------------------- A
A_rows = []
for pos in np.arange(5.0, 6.01, 0.1):
    base_c = S.template_centroid(template(10, 0.15, 0.3, pos, 1.5))
    for s in np.arange(-0.6, 0.61, 0.1):
        T = template(10, 0.15, 0.3, pos + s, 1.5)
        cen_corr = S.template_centroid(T) - s                       # centroid minus true drift
        reg = S.template_centroid(S.register_spikes(T[None], [s])[0])   # registered back by interpolation
        A_rows.append(dict(pos=round(pos, 2), shift=round(s, 2), centroid_minus_drift_err=cen_corr - base_c,
                           registered_err=reg - base_c))
e1 = np.array([r['centroid_minus_drift_err'] for r in A_rows]); e2 = np.array([r['registered_err'] for r in A_rows])
out['A'] = dict(centroid_minus_drift=dict(max_abs=round(float(np.abs(e1).max()), 4), rms=round(float(np.sqrt((e1 ** 2).mean())), 4)),
                registered=dict(max_abs=round(float(np.abs(e2).max()), 4), rms=round(float(np.sqrt((e2 ** 2).mean())), 4)),
                centroid_vs_true_slope=round(float(np.polyfit([r['shift'] for r in A_rows if r['pos'] == 5.5],
                                                               [S.template_centroid(template(10, .15, .3, 5.5 + r['shift'], 1.5)) for r in A_rows if r['pos'] == 5.5], 1)[0]), 3))
print('A', json.dumps(out['A']))


# ---------------------------------------------------------------- B
def noise(rng, n):
    white = rng.standard_normal((n, 48 + 1, N_CH + 2)).astype(np.float32)
    x = 0.25 * white[:, :, :-2] + 0.5 * white[:, :, 1:-1] + 0.25 * white[:, :, 2:]
    x = 0.6 * x[:, 1:] + 0.4 * x[:, :-1]
    return x / x.std()


def one_neuron(rng, n, amp, drift):
    t = np.sort(rng.uniform(0, 60 * FS, n))
    pos = 7.3 + drift * t / (60 * FS)
    S_ = np.array([template(amp * (1 + 0.07 * rng.standard_normal()), 0.15, 0.3, p, 1.5) for p in pos]) + noise(rng, n)
    return t, S_


def trial(seed, amp, drift, n=400, corr='none'):
    rng = np.random.default_rng(seed)
    t, Sn = one_neuron(rng, n, amp, drift)
    lab, test = S._forced_split(Sn.reshape(n, -1), seed)
    tp, tq = np.where((lab == 0) & test)[0], np.where((lab == 1) & test)[0]
    if min(len(tp), len(tq)) < S.MIN_N // 2:
        return None
    dfn = (lambda x: drift * np.asarray(x) / (60 * FS)) if corr != 'none' else None
    if corr == 'none':
        gap, se = S.template_gap(Sn[tp], Sn[tq], seed)
    else:
        gap, se = S.general_gap(t[tp], Sn[tp], t[tq], Sn[tq], 'naive', dfn, seed, register=(corr == 'registered'))
    return gap, gap / max(se, 1e-9)


B = {}
for amp in (7.0, 12.0):
    for drift, corr in ((0.0, 'none'), (0.6, 'none'), (0.6, 'centroid'), (0.6, 'registered')):
        res = [r for r in (trial(1000 + k, amp, drift, corr=corr) for k in range(150)) if r is not None]
        g = np.array([r[0] for r in res]); z = np.array([r[1] for r in res])
        key = f'amp{int(amp)}_drift{drift}_{corr}'
        B[key] = dict(trials=len(res), median_gap=round(float(np.median(g)), 3), p95_gap=round(float(np.percentile(g, 95)), 3),
                      false_reopen_rate=round(float(np.mean((g >= S.POS_MIN) & (z >= S.POS_Z))), 3),
                      z_ge_3_rate=round(float(np.mean(z >= S.POS_Z)), 3))
        print('B', key, B[key], flush=True)
out['B'] = B
json.dump(out, open('results/null_split.json', 'w'), indent=1)
