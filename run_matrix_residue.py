"""Protocol R1: what the best matrix neuron cannot explain.

    python run_matrix_residue.py calibration   # seeds 900-902, for debugging and tuning only
    python run_matrix_residue.py test          # seeds 1000-1009, run once on frozen code

Each seed is simulated as three truths (T0 passive, T1 passive + tanh readout, T2 NMDA) and
every truth is observed by the matrix neuron (lin) and the best fixed-filter cascade (ln).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

from sihtisort import cable as K
from sihtisort import observer as O

SETS = {'calibration': [900, 901, 902], 'test': list(range(1000, 1010))}
ARMS = [('T0', 'lin'), ('T1', 'lin'), ('T2', 'lin'), ('T0', 'ln'), ('T1', 'ln'), ('T2', 'ln')]
N_NULL = 20
N_BOOT = 2000


def run_seed(seed):
    t0 = time.time()
    c = K.make_cell(seed)
    Xr = O.bank(c['counts'])
    W = K.windows()
    a, b = W['train']
    X = O.Standardizer(Xr[a:b])(Xr)
    cols = O.branch_cols(c['site_branch'])
    gen = O.GenProbe(X, W)
    out = dict(morph=c['morph'], n_bursts=len(c['bursts']))
    for truth, model in ARMS:
        r = O.run_arm(model, X, c['y'][truth], W, cols, gen, n_null=N_NULL, seed=seed,
                      y_clean=c['y_clean'][truth])
        out[f'{truth}_{model}'] = r
        print(f'seed {seed} {truth} {model}: S gen {r["S"]["gen"]:+.4f} (bar {r["cal_gen"]["bar"]:.4f})'
              f'  state {r["S"]["state"]:+.4f} (bar {r["cal_state"]["bar"]:.4f})'
              f'  R2 hold {r["r2_hold"]:.3f} B {r["r2_B"]:.3f}  ({time.time() - t0:.0f} s)', flush=True)
    return seed, out


def boot_ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(res):
    rng = np.random.default_rng(0)
    seeds = sorted(res)
    n = len(seeds)
    arm = lambda t, m: [res[s][f'{t}_{m}'] for s in seeds]
    count = lambda t, m, k='cal_gen': int(sum(r[k]['structured'] for r in arm(t, m)))
    table = {}
    for t, m in ARMS:
        rs = arm(t, m)
        table[f'{t}_{m}'] = dict(
            structured_gen=count(t, m), structured_state=count(t, m, 'cal_state'),
            naive_structured_gen=count(t, m, 'naive_gen'), naive_structured_state=count(t, m, 'naive_state'),
            S_gen=[round(r['S']['gen'], 4) for r in rs], S_state=[round(r['S']['state'], 4) for r in rs],
            bar_gen=[round(r['cal_gen']['bar'], 4) for r in rs], bar_state=[round(r['cal_state']['bar'], 4) for r in rs],
            r2_hold=float(np.mean([r['r2_hold'] for r in rs])), r2_B=float(np.mean([r['r2_B'] for r in rs])),
            r2_B_plus_state=float(np.mean([r['r2_B_plus']['state'] for r in rs])),
            r2_B_plus_gen=float(np.mean([r['r2_B_plus']['gen'] for r in rs])),
            ceiling_hold=float(np.mean([r['ceiling_hold'] for r in rs])),
            ceiling_B=float(np.mean([r['ceiling_B'] for r in rs])))
    d2 = np.array([r['r2_B_plus']['state'] - r['r2_B'] for r in arm('T2', 'ln')])
    d1 = np.array([r['r2_B_plus']['state'] - r['r2_B'] for r in arm('T1', 'ln')])
    g = {}
    g['R0'] = dict(T0_lin_structured=table['T0_lin']['structured_gen'],
                   T1_ln_structured=table['T1_ln']['structured_gen'], n=n,
                   passed=table['T0_lin']['structured_gen'] <= 1 and table['T1_ln']['structured_gen'] <= 1)
    g['R1'] = dict(T1_lin_structured=table['T1_lin']['structured_gen'],
                   T2_lin_structured=table['T2_lin']['structured_gen'], n=n,
                   passed=table['T1_lin']['structured_gen'] >= 8 and table['T2_lin']['structured_gen'] >= 8)
    g['R2'] = dict(T2_ln_structured=table['T2_ln']['structured_gen'], n=n,
                   passed=table['T2_ln']['structured_gen'] >= 8)
    ci2, cid = boot_ci(d2, rng), boot_ci(d2 - d1, rng)
    g['R3'] = dict(delta_T2=[round(float(x), 4) for x in d2], delta_T1=[round(float(x), 4) for x in d1],
                   mean_T2=float(d2.mean()), ci_T2=ci2, mean_T2_minus_T1=float((d2 - d1).mean()), ci_T2_minus_T1=cid,
                   passed=bool(d2.mean() > 0 and ci2[0] > 0 and cid[0] > 0))
    # reported: how much of regime B each model misses, relative to the noise-free truth
    for t in ('T1', 'T2'):
        for m in ('lin', 'ln'):
            gap = np.array([r['ceiling_B'] - r['r2_B'] for r in arm(t, m)])
            g.setdefault('B_gap', {})[f'{t}_{m}'] = dict(mean=float(gap.mean()), ci=boot_ci(gap, rng))
    gap1 = np.array([r['ceiling_B'] - r['r2_B'] for r in arm('T1', 'ln')])
    gap2 = np.array([r['ceiling_B'] - r['r2_B'] for r in arm('T2', 'ln')])
    g['B_gap']['T2_ln_minus_T1_ln'] = dict(mean=float((gap2 - gap1).mean()), ci=boot_ci(gap2 - gap1, rng))
    return table, g


def main(which):
    t0 = time.time()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', 'sihtisort', 'run_matrix_residue.py'],
                                capture_output=True, text=True).stdout.strip())
    with Pool(2) as pool:
        res = dict(pool.map(run_seed, SETS[which]))
    table, g = summarize(res)
    out = dict(set=which, seeds=SETS[which], commit=commit, code_dirty=dirty, n_null=N_NULL,
               params=K.PARAMS, table=table, gates=g, per_seed={str(s): res[s] for s in sorted(res)},
               minutes=round((time.time() - t0) / 60, 1))
    os.makedirs('results', exist_ok=True)
    with open(f'results/matrix_residue_{which}.json', 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    for k, v in table.items():
        print(f"{k:7s} structured gen {v['structured_gen']}/{len(SETS[which])} state {v['structured_state']}"
              f"  (naive {v['naive_structured_gen']}/{v['naive_structured_state']})"
              f"  R2 hold {v['r2_hold']:.3f}/{v['ceiling_hold']:.3f}  B {v['r2_B']:.3f}/{v['ceiling_B']:.3f}"
              f"  B+state {v['r2_B_plus_state']:.3f}")
    for k, v in g.items():
        print(k, v)
    print(f'{(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'calibration')
