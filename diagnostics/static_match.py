"""POST HOC (after the Protocol R1 fresh run): is the cascade's regime-B gap on the NMDA neuron
about the operator, or only about how far regime B's amplitudes are from regime A's?

Two truths that are a fixed filter followed by a static nonlinearity (inside the cascade's class)
but reach T2's amplitudes in regime B:

  T1x-iso  the monotone map that best carries the passive soma onto the NMDA soma (isotonic
           regression over the whole noise-free recording). Tried first; see the README.
  T1x-exp  an expansive exponential, v -> s (exp(v/s) - 1), with s set per cell so that the
           99.9th percentile of the noise-free regime-B depolarization equals T2's.

If the cascade misses regime B of these about as badly as T2, the gap is extrapolation in
amplitude. If it misses them much less, the gap belongs to the state-dependent operator.

    python diagnostics/static_match.py     # seeds 1000-1009 (~8 min on 2 cores)
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.isotonic import IsotonicRegression

from sihtisort import cable as K
from sihtisort import observer as O

SEEDS = list(range(1000, 1010))


def run(seed):
    c = K.make_cell(seed)
    Xr = O.bank(c['counts'])
    W = K.windows()
    a, b = W['train']
    X = O.Standardizer(Xr[a:b])(Xr)
    (a, b), (e, f) = W['train'], W['B']
    # the maps are fitted noise-free; the recorded T0 (intrinsic and measurement noise included)
    # is then pushed through them, so here the measurement noise also passes through the map
    c0, c2 = c['y_clean']['T0'] - K.E_L, c['y_clean']['T2'] - K.E_L
    v0 = c['y']['T0'] - K.E_L
    iso = IsotonicRegression(out_of_bounds='clip').fit(c0, c2)
    target = np.percentile(c2[e:f], 99.9)
    p0 = np.percentile(c0[e:f], 99.9)
    lo, hi = 0.05, 100.0                    # s (mV): bisect so that s (exp(p0/s) - 1) = target
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if mid * np.expm1(p0 / mid) > target else (lo, mid)
    s_exp = float(np.sqrt(lo * hi))
    maps = {'iso': iso.predict, 'exp': lambda v: s_exp * np.expm1(v / s_exp)}
    out = dict(seed=seed, map_r2_on_T2_clean=O.r2(c2, iso.predict(c0)), s_exp=s_exp,
               T2_B_p999=float(target))
    for mname, g in maps.items():
        y_clean = K.E_L + g(c0)
        y = K.E_L + g(v0)
        r = dict(A_range_mV=float(np.ptp(y_clean[a:b])), B_range_mV=float(np.ptp(y_clean[e:f])))
        for name in ('lin', 'ln'):
            lin = O.MatrixNeuron().fit(X[a:b], y[a:b])
            m = lin if name == 'lin' else O.Cascade().fit(X[a:b], y[a:b], lin=lin)
            P = m.predict(X)
            r[name] = dict(r2_B=O.r2(y[e:f], P[e:f]), ceiling_B=O.r2(y[e:f], y_clean[e:f]))
            r[name]['gap'] = r[name]['ceiling_B'] - r[name]['r2_B']
        out[mname] = r
    print(seed, {m: (round(out[m]['lin']['gap'], 3), round(out[m]['ln']['gap'], 3)) for m in maps}, flush=True)
    return out


def main():
    with Pool(2) as p:
        res = p.map(run, SEEDS)
    main_run = json.load(open('results/matrix_residue_test.json'))
    rng = np.random.default_rng(0)
    boot = lambda x: [float(np.percentile(x[rng.integers(0, len(x), (2000, len(x)))].mean(1), q)) for q in (2.5, 97.5)]
    gap = lambda t: np.array([main_run['per_seed'][str(r['seed'])][f'{t}_ln']['ceiling_B']
                              - main_run['per_seed'][str(r['seed'])][f'{t}_ln']['r2_B'] for r in res])
    g1, g2 = gap('T1'), gap('T2')
    summary = dict(cascade_B_gap=dict(T1=dict(mean=float(g1.mean()), median=float(np.median(g1))),
                                      T2=dict(mean=float(g2.mean()), median=float(np.median(g2)))))
    for m in ('iso', 'exp'):
        gx = np.array([r[m]['ln']['gap'] for r in res])
        gl = np.array([r[m]['lin']['gap'] for r in res])
        summary['cascade_B_gap'][f'T1x_{m}'] = dict(mean=float(gx.mean()), median=float(np.median(gx)), ci=boot(gx),
                                                   per_seed=[round(float(v), 4) for v in gx])
        summary[f'matrix_B_gap_T1x_{m}'] = dict(mean=float(gl.mean()), median=float(np.median(gl)))
        summary[f'T2_minus_T1x_{m}'] = dict(mean=float((g2 - gx).mean()), median=float(np.median(g2 - gx)),
                                            ci=boot(g2 - gx), per_seed=[round(float(v), 4) for v in g2 - gx])
    json.dump(dict(post_hoc=True, per_seed=res, summary=summary), open('results/static_match.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
