"""PROTOCOL_6: register each spike before averaging (Sol).

    python diagnostics/register_first.py calib     # seeds 92-94, debugging only
    python diagnostics/register_first.py           # fresh sets: linear 170-181, sine 190-201, no drift 210-221
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort import sort as S                          # noqa: E402
from sihtisort.evaluate import global_match, score       # noqa: E402
from sihtisort.synth import drift_at, make_recording     # noqa: E402

METHODS = ['frame-oracle-centroid', 'frame-oracle-registered', 'frame-estimated-centroid',
           'frame-estimated-registered', 'frame-wrong-registered', 'switch']
SETS = {'linear': (list(range(170, 182)), dict(drift=0.6, dwell=20.0)),
        'sine': (list(range(190, 202)), dict(drift=0.3, dwell=20.0, drift_kind='sine')),
        'no_drift': (list(range(210, 222)), {})}
CALIB = {'calib_linear': ([92, 93], dict(drift=0.6, dwell=20.0)),
         'calib_sine': ([94], dict(drift=0.3, dwell=20.0, drift_kind='sine'))}


def counts(sc):
    one = lambda k: sum(r['one_unit'] for r in sc['neurons'] if r['kind'].startswith(k))
    return dict(mood=one('mood'), stable=one('stable'), turn_merged=sc['turn_merged'], false_merges=sc['false_merges'])


def reopen_truth(t, before, after, rem, gt):
    gm = global_match(gt['t'], t)
    inv = np.full(len(t), -1); ok = gm >= 0; inv[gm[ok]] = gt['unit'][ok]

    def dom(mask):
        ids = inv[mask]; ids = ids[ids >= 0]
        return int(np.bincount(ids).argmax()) if len(ids) else None
    res = []
    for r in rem:
        if r['status'] != 'reopened':
            continue
        moved = after == r['new_unit']
        stayed = (before == r['address']) & ~moved
        a, b = dom(stayed), dom(moved)
        res.append(dict(kind=r['kind'], by=r['by'], gap=r['position_gap'], false=(a == b)))
    return res


def run_seed(args):
    seed, kw = args
    data, gt, meta = make_recording(seed, min_sep=1.0, turn_range=(0.0, 0.5), **kw)
    base = S.snapshot_sort(data, 0)
    out = S.run_s(data, base=base, commit=True, two_channel=True)
    t, addr, st, clean = out['_rem_inputs']
    assert np.all(np.diff(t) >= 0)
    est = S.estimate_drift(t, addr, clean)
    Tend = float(t.max())
    oracle = lambda x: drift_at(meta, x)
    wrong = (lambda x: est(Tend - np.asarray(x, float))) if est is not None else None
    fns = {'frame-oracle-centroid': ('naive', oracle, False), 'frame-oracle-registered': ('naive', oracle, True),
           'frame-estimated-centroid': ('naive', est, False), 'frame-estimated-registered': ('naive', est, True),
           'frame-wrong-registered': ('naive', wrong, True), 'switch': ('switch', None, False)}
    ta = [n['rate'] for n in meta['neurons'] if n['kind'] == 'turn_a'][0]
    tb = [n['rate'] for n in meta['neurons'] if n['kind'] == 'turn_b'][0]
    rec = dict(seed=seed, drift=meta['drift'], drift_kind=meta['drift_kind'], turn_offset=round(meta['turn_offset'], 3),
               turn_rate_ratio=round(max(ta, tb) / min(ta, tb), 3),
               committed=counts(score(dict(t=t, unit=addr, state=st), gt, meta)))
    grid = np.linspace(0, Tend, 50)
    if est is not None:
        true = drift_at(meta, grid); e = est(grid)
        rec['drift_estimate_rms_error'] = round(float(np.sqrt(np.mean(((true - true.mean()) - (e - e.mean())) ** 2))), 4)
        rec['true_drift_range'] = round(float(np.ptp(true)), 3)
    for name, (mode, fn, reg) in fns.items():
        if fn is None and name != 'switch':
            rec[name] = None
            continue
        units, rem = S.remainder_channel(t, addr, st, clean, seed, mode=mode, drift_fn=fn, register=reg)
        rec[name] = dict(counts=counts(score(dict(t=t, unit=units, state=st), gt, meta)),
                         reopenings=reopen_truth(t, addr, units, rem, gt))
        if name == 'switch':
            rec[name]['spans_s'] = [x for r in rem for x in r.get('switch_spans_s', [])]
    return rec


def judge(rows, m):
    sep = [r for r in rows if r['committed']['turn_merged'] and (r['turn_offset'] >= 0.25 or r['turn_rate_ratio'] >= 1.5)]
    rr = [r for r in rows if r[m] is not None]
    fr = sum(x['false'] for r in rr for x in r[m]['reopenings'])
    tr = sum(not x['false'] for r in rr for x in r[m]['reopenings'])
    cm = sum(r['committed']['mood'] for r in rr); mm = sum(r[m]['counts']['mood'] for r in rr)
    cs = sum(r['committed']['stable'] for r in rr); ms = sum(r[m]['counts']['stable'] for r in rr)
    sp = [r for r in sep if r[m] is not None]
    separated = sum(not r[m]['counts']['turn_merged'] for r in sp)
    ok_sep = (separated / len(sp) >= 0.7) if sp else None
    passed = fr <= 1 and mm >= cm - 1 and ms >= cs - 2 and (ok_sep is not False)
    return dict(false_reopenings=fr, true_reopenings=tr, mood=(cm, mm), stable=(cs, ms),
                separable=len(sp), separated=separated, recordings=len(rr), passed=passed)


if __name__ == '__main__':
    calib = len(sys.argv) > 1 and sys.argv[1] == 'calib'
    sets = CALIB if calib else SETS
    report = {}
    for name, (seeds, kw) in sets.items():
        with Pool(2) as pool:
            rows = pool.map(run_seed, [(s, kw) for s in seeds])
        table = {m: judge(rows, m) for m in METHODS}
        report[name] = dict(table=table, rows=rows)
        print(name, flush=True)
        for m in METHODS:
            print(f'   {m:16s}', table[m], flush=True)
        print('   drift estimate rms error:', [(r['seed'], r.get('drift_estimate_rms_error'), r.get('true_drift_range')) for r in rows], flush=True)
        spans = [x for r in rows for x in r['switch']['spans_s']]
        if spans:
            report[name]['switch_spans_s'] = dict(n=len(spans), median=round(float(np.median(spans)), 2),
                                                  p95=round(float(np.percentile(spans, 95)), 2), max=round(float(max(spans)), 2))
            print('   switch spans (s):', report[name]['switch_spans_s'], flush=True)
    if not calib:
        T = {k: report[k]['table'] for k in report}
        report['gates'] = dict(
            G1=dict(registered_sine=T['sine']['frame-oracle-registered']['false_reopenings'],
                    centroid_sine=T['sine']['frame-oracle-centroid']['false_reopenings'],
                    passed=T['sine']['frame-oracle-registered']['passed']
                    and T['sine']['frame-oracle-registered']['false_reopenings'] < T['sine']['frame-oracle-centroid']['false_reopenings']),
            G2=all(T[k]['frame-estimated-registered']['passed'] for k in T),
            G3=dict(wrong_registered_false_reopenings_linear=T['linear']['frame-wrong-registered']['false_reopenings'],
                    control_behaves=T['linear']['frame-wrong-registered']['false_reopenings'] >= 3))
        print('GATES', report['gates'], flush=True)
    out = 'results/register_first_calib.json' if calib else 'results/register_first.json'
    json.dump(report, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
