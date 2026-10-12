"""Protocol R2: where does the state enter? The two-site gain invariant (PROTOCOL_R2.md).

    python run_probe_kernels.py calibration   # seeds 920-922, for debugging and tuning only
    python run_probe_kernels.py test          # seeds 2000-2009, run once on frozen code
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

from sihtisort import observer as O
from sihtisort import probe as PR

SETS = {'calibration': [920, 921, 922], 'test': list(range(2000, 2010))}
TRUTHS = ['T0', 'T1', 'T1f', 'T2']
N_BOOT = 2000


def run_seed(seed):
    t0 = time.time()
    c = PR.make_probed_cell(seed)
    Xr = O.bank(c['counts'])
    a = int(round(500.0 / 2.0))
    X = O.Standardizer(Xr[a:])(Xr)
    out = dict(T1_scale=c['s'], eps=c['eps'])
    for truth in TRUTHS:
        for label, state in (('noisy', 'branch'), ('noisy', 'soma'), ('clean', 'branch')):
            r = PR.analyse(c['y'][label][truth], c['xi'], c['site_branch'], c['eps'], X,
                           xi_ref=c['xi_ref'], seed=seed, state=state)
            out[f'{truth}_{label}_{state}'] = r
            st = r['stat']
            print(f'seed {seed} {truth:3s} {label} {state:6s}: z all {st["all"]["z"]:+.1f} same {st["same"]["z"]:+.1f} '
                  f'cross {st["cross"]["z"]:+.1f}  rejected {st["all"]["rejected"]}  snr {min(r["snr"]):.0f}'
                  f'  ({time.time() - t0:.0f} s)', flush=True)
    return seed, out


def boot_ci(x, rng):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(res):
    rng = np.random.default_rng(0)
    seeds = sorted(res)
    n = len(seeds)
    table = {}
    for truth in TRUTHS:
        for cond in ('noisy_branch', 'noisy_soma', 'clean_branch'):
            rs = [res[s][f'{truth}_{cond}'] for s in seeds]
            row = {}
            for part in ('all', 'same', 'cross'):
                z = [r['stat'][part]['z'] for r in rs]
                row[part] = dict(rejected=int(sum(r['stat'][part]['rejected'] for r in rs)),
                                 z=[round(v, 2) for v in z], z_mean=float(np.mean(z)), z_ci=boot_ci(z, rng))
            row['decay_ms_per_bin_mean'] = np.mean([r['decay_ms_per_bin'] for r in rs], 0).round(2).tolist()
            row['snr_min'] = float(min(min(r['snr']) for r in rs))
            table[f'{truth}_{cond}'] = row
    g = {}
    g['Q0'] = dict(T0_rejected=table['T0_noisy_branch']['all']['rejected'],
                   T1_rejected=table['T1_noisy_branch']['all']['rejected'], n=n,
                   passed=table['T0_noisy_branch']['all']['rejected'] <= 1
                   and table['T1_noisy_branch']['all']['rejected'] <= 1)
    g['Q1'] = dict(T2_rejected=table['T2_noisy_branch']['all']['rejected'], n=n,
                   passed=table['T2_noisy_branch']['all']['rejected'] >= 8)
    z2 = np.array(table['T2_noisy_branch']['all']['z'])
    z1f = np.array(table['T1f_noisy_branch']['all']['z'])
    g['reported_T2_minus_T1f_z'] = dict(mean=float((z2 - z1f).mean()), ci=boot_ci(z2 - z1f, rng))
    return table, g


def main(which):
    t0 = time.time()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', 'sihtisort', 'run_probe_kernels.py'],
                                capture_output=True, text=True).stdout.strip())
    with Pool(2) as pool:
        res = dict(pool.map(run_seed, SETS[which], chunksize=1))
    table, g = summarize(res)
    out = dict(set=which, seeds=SETS[which], commit=commit, code_dirty=dirty,
               settings=dict(eps=PR.EPS, duration_ms=PR.DURATION, state=PR.STATE, n_perm=PR.N_PERM,
                             n_bins=PR.N_BINS, lags=PR.LAGS, window_frac=PR.WINDOW_FRAC, t1f_tau=PR.T1F_TAU),
               table=table, gates=g, per_seed={str(s): res[s] for s in sorted(res)},
               minutes=round((time.time() - t0) / 60, 1))
    os.makedirs('results', exist_ok=True)
    with open(f'results/probe_kernels_{which}.json', 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    for k, v in table.items():
        print(f"{k:18s} rejected all {v['all']['rejected']}/{len(SETS[which])} same {v['same']['rejected']} "
              f"cross {v['cross']['rejected']}  mean z all {v['all']['z_mean']:+.2f} same {v['same']['z_mean']:+.2f} "
              f"cross {v['cross']['z_mean']:+.2f}  snr>={v['snr_min']:.0f}")
    for k, v in g.items():
        print(k, v)
    print(f'{(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'calibration')
