"""PROTOCOL_7: judge each forced split against a calibrated one-neuron null (Sol).

    python diagnostics/calibrated.py calib     # seeds 92-94, debugging only
    python diagnostics/calibrated.py           # fresh sets: linear 230-249, sine 250-269, no drift 270-289
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort import sort as S                          # noqa: E402
from sihtisort.evaluate import score                     # noqa: E402
from sihtisort.synth import drift_at, make_recording     # noqa: E402
from register_first import counts, reopen_truth          # noqa: E402

VARIANTS = ['uncal-registered', 'cal-registered', 'cal-uncorrected']
SETS = {'linear': (list(range(230, 250)), dict(drift=0.6, dwell=20.0)),
        'sine': (list(range(250, 270)), dict(drift=0.3, dwell=20.0, drift_kind='sine')),
        'no_drift': (list(range(270, 290)), {})}
CALIB = {'calib_linear': ([92, 93], dict(drift=0.6, dwell=20.0)),
         'calib_sine': ([94], dict(drift=0.3, dwell=20.0, drift_kind='sine'))}
BOOT = 2000


def run_seed(args):
    seed, kw = args
    t0 = time.time()
    data, gt, meta = make_recording(seed, min_sep=1.0, turn_range=(0.0, 0.5), **kw)
    base = S.snapshot_sort(data, 0)
    out = S.run_s(data, base=base, commit=True, two_channel=True)
    t, addr, st, clean = out['_rem_inputs']
    assert np.all(np.diff(t) >= 0)
    est = S.estimate_drift(t, addr, clean)
    cfg = {'uncal-registered': dict(drift_fn=est, register=True),
           'cal-registered': dict(drift_fn=est, register=True, calibrate=True, null_drift=est),
           'cal-uncorrected': dict(drift_fn=None, calibrate=True, null_drift=est)}
    ta = [n['rate'] for n in meta['neurons'] if n['kind'] == 'turn_a'][0]
    tb = [n['rate'] for n in meta['neurons'] if n['kind'] == 'turn_b'][0]
    rec = dict(seed=seed, drift=meta['drift'], drift_kind=meta['drift_kind'], turn_offset=round(meta['turn_offset'], 3),
               turn_rate_ratio=round(max(ta, tb) / min(ta, tb), 3), drift_estimated=est is not None,
               committed=counts(score(dict(t=t, unit=addr, state=st), gt, meta)))
    if est is not None:
        grid = np.linspace(0, float(t.max()), 50)
        true = drift_at(meta, grid); e = est(grid)
        rec['drift_estimate_rms_error'] = round(float(np.sqrt(np.mean(((true - true.mean()) - (e - e.mean())) ** 2))), 4)
    for name, kw2 in cfg.items():
        units, rem = S.remainder_channel(t, addr, st, clean, seed, mode='naive', **kw2)
        ro = reopen_truth(t, addr, units, rem, gt)
        rec[name] = dict(counts=counts(score(dict(t=t, unit=units, state=st), gt, meta)), reopenings=ro,
                         forced_splits=[dict(address=r['address'], gap=r.get('position_gap'), z=r.get('position_z'),
                                             rate_z=(r.get('rate') or {}).get('z'), status=r['status'], by=r.get('by'),
                                             null=r.get('null'))
                                        for r in rem if r.get('kind') == 'forced split'])
    rec['seconds'] = round(time.time() - t0, 1)
    return rec


def separable(r):
    return r['committed']['turn_merged'] and (r['turn_offset'] >= 0.25 or r['turn_rate_ratio'] >= 1.5)


def false_n(r, v):
    return sum(x['false'] for x in r[v]['reopenings'])


def judge(rows, v):
    fr = sum(false_n(r, v) for r in rows)
    tr = sum(not x['false'] for r in rows for x in r[v]['reopenings'])
    cm = sum(r['committed']['mood'] for r in rows); mm = sum(r[v]['counts']['mood'] for r in rows)
    cs = sum(r['committed']['stable'] for r in rows); ms = sum(r[v]['counts']['stable'] for r in rows)
    sp = [r for r in rows if separable(r)]
    separated = sum(not r[v]['counts']['turn_merged'] for r in sp)
    return dict(false_reopenings=fr, true_reopenings=tr, mood=(cm, mm), stable=(cs, ms), mood_drop=cm - mm,
                stable_drop=cs - ms, separable=len(sp), separated=separated, recordings=len(rows),
                safe=fr <= 2 and cm - mm <= 1 and cs - ms <= 2)


def ci(vals, seed=0):
    """Mean over recordings with a 95% recording-bootstrap interval."""
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = vals[rng.integers(0, len(vals), (BOOT, len(vals)))].mean(1)
    return dict(mean=round(float(vals.mean()), 3), lo=round(float(np.percentile(bs, 2.5)), 3),
                hi=round(float(np.percentile(bs, 97.5)), 3), n=len(vals))


def pooled(rows):
    P = {}
    for v in VARIANTS:
        P[v] = dict(false_per_recording=ci([false_n(r, v) for r in rows]),
                    recall=ci([float(not r[v]['counts']['turn_merged']) for r in rows if separable(r)]))
    P['diff_cal_minus_uncal'] = ci([false_n(r, 'cal-registered') - false_n(r, 'uncal-registered') for r in rows])
    return P


if __name__ == '__main__':
    calib = len(sys.argv) > 1 and sys.argv[1] == 'calib'
    sets = CALIB if calib else SETS
    report, allrows = {}, []
    for name, (seeds, kw) in sets.items():
        with Pool(2) as pool:
            rows = pool.map(run_seed, [(s, kw) for s in seeds])
        allrows += rows
        table = {v: judge(rows, v) for v in VARIANTS}
        report[name] = dict(table=table, rows=rows)
        print(name, flush=True)
        for v in VARIANTS:
            print(f'   {v:17s}', table[v], flush=True)
        print('   seconds:', [r['seconds'] for r in rows], flush=True)
    P = pooled(allrows)
    report['pooled'] = P
    print('POOLED', json.dumps(P), flush=True)
    if not calib:
        T = {k: report[k]['table'] for k in SETS}
        d = P['diff_cal_minus_uncal']
        tot = lambda v: sum(T[k][v]['false_reopenings'] for k in T)
        sep = sum(T[k]['cal-registered']['separable'] for k in T)
        sepd = sum(T[k]['cal-registered']['separated'] for k in T)
        report['gates'] = dict(
            C1=dict(per_set={k: T[k]['cal-registered']['safe'] for k in T}, passed=all(T[k]['cal-registered']['safe'] for k in T)),
            C2=dict(cal=tot('cal-registered'), uncal=tot('uncal-registered'), diff_ci=d,
                    passed=tot('cal-registered') < tot('uncal-registered') and d['hi'] < 0),
            C3=dict(separable=sep, separated=sepd, recall=round(sepd / sep, 3) if sep else None,
                    passed=(sepd / sep >= 0.7) if sep else None))
        print('GATES', json.dumps(report['gates']), flush=True)
    out = 'results/calibrated_calib.json' if calib else 'results/calibrated.json'
    json.dump(report, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
