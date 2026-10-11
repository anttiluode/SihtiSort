"""PROTOCOL_4: the drift control (Sol's suggestion).

    python diagnostics/drift_control.py calib     # seeds 92-93, debugging only
    python diagnostics/drift_control.py           # drift set 80-91, then no-drift regression 50-61
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort import sort as S                         # noqa: E402
from sihtisort.evaluate import global_match, score      # noqa: E402
from sihtisort.synth import make_recording              # noqa: E402

DRIFT = dict(drift=0.6, dwell=20.0)


def counts(sc):
    one = lambda k: sum(r['one_unit'] for r in sc['neurons'] if r['kind'].startswith(k))
    return dict(mood=one('mood'), stable=one('stable'), turn_merged=sc['turn_merged'], false_merges=sc['false_merges'])


def reopen_truth(out, units_key, rem_key, gt):
    """For every reopening, is it false (both parts dominated by the same true neuron)?"""
    o = np.argsort(out['t'], kind='stable'); t = out['t'][o]
    before, after = out['unit'][o], out[units_key][o]
    gm = global_match(gt['t'], t)
    inv = np.full(len(t), -1); ok = gm >= 0; inv[gm[ok]] = gt['unit'][ok]

    def dom(mask):
        ids = inv[mask]; ids = ids[ids >= 0]
        return int(np.bincount(ids).argmax()) if len(ids) else None
    res = []
    for r in out[rem_key]:
        if r['status'] != 'reopened':
            continue
        moved = after == r['new_unit']
        stayed = (before == r['address']) & ~moved
        a, b = dom(stayed), dom(moved)
        res.append(dict(kind=r['kind'], by=r['by'], gap=r['position_gap'], true_a=a, true_b=b, false=(a == b)))
    return res


def run_seed(args):
    seed, kw = args
    data, gt, meta = make_recording(seed, min_sep=1.0, turn_range=(0.0, 0.5), **kw)
    base = S.snapshot_sort(data, 0)
    out = S.run_s(data, base=base, commit=True, two_channel=True)
    ta = [n['rate'] for n in meta['neurons'] if n['kind'] == 'turn_a'][0]
    tb = [n['rate'] for n in meta['neurons'] if n['kind'] == 'turn_b'][0]
    sc = lambda key: counts(score(dict(t=out['t'], unit=out[key], state=out['state']), gt, meta))
    return dict(seed=seed, drift=meta['drift'], turn_offset=round(meta['turn_offset'], 3),
                turn_rate_ratio=round(max(ta, tb) / min(ta, tb), 3),
                committed=sc('unit'), naive=sc('unit_two_channel'), aware=sc('unit_two_channel_aware'),
                reopen_naive=reopen_truth(out, 'unit_two_channel', 'remainder', gt),
                reopen_aware=reopen_truth(out, 'unit_two_channel_aware', 'remainder_aware', gt))


def summary(rows):
    tot = lambda v, k: sum(r[v][k] for r in rows)
    sep = [r for r in rows if r['committed']['turn_merged'] and (r['turn_offset'] >= 0.25 or r['turn_rate_ratio'] >= 1.5)]
    out = {}
    for v in ('naive', 'aware'):
        fr = sum(x['false'] for r in rows for x in r[f'reopen_{v}'])
        out[v] = dict(false_reopenings=fr, true_reopenings=sum(not x['false'] for r in rows for x in r[f'reopen_{v}']),
                      mood=tot(v, 'mood'), stable=tot(v, 'stable'), false_merges=tot(v, 'false_merges'),
                      separable_pairs_merged_by_commit=len(sep), separated=sum(not r[v]['turn_merged'] for r in sep))
    out['committed'] = dict(mood=tot('committed', 'mood'), stable=tot('committed', 'stable'),
                            false_merges=tot('committed', 'false_merges'))
    return out


def gates_drift(s):
    c, n, a = s['committed'], s['naive'], s['aware']
    return dict(
        D0=dict(naive_false_reopenings=n['false_reopenings'], danger_real=n['false_reopenings'] >= 3),
        D1=dict(aware_false_reopenings=a['false_reopenings'], mood=(c['mood'], a['mood']), stable=(c['stable'], a['stable']),
                passed=a['false_reopenings'] <= 1 and a['mood'] >= c['mood'] - 1 and a['stable'] >= c['stable'] - 2),
        D2=dict(separable=a['separable_pairs_merged_by_commit'], separated=a['separated'],
                passed=a['separable_pairs_merged_by_commit'] > 0 and a['separated'] / a['separable_pairs_merged_by_commit'] >= 0.7))


def gates_nodrift(s):
    c, a = s['committed'], s['aware']
    return dict(D3=dict(separable=a['separable_pairs_merged_by_commit'], separated=a['separated'],
                        naive_separated=s['naive']['separated'], mood=(c['mood'], a['mood']), stable=(c['stable'], a['stable']),
                        aware_false_reopenings=a['false_reopenings'],
                        passed=(a['separable_pairs_merged_by_commit'] > 0
                                and a['separated'] / a['separable_pairs_merged_by_commit'] >= 0.7
                                and a['mood'] >= c['mood'] - 1 and a['stable'] >= c['stable'] - 2)))


if __name__ == '__main__':
    calib = len(sys.argv) > 1 and sys.argv[1] == 'calib'
    sets = {'calib': ([92, 93], DRIFT)} if calib else {'drift': (list(range(80, 92)), DRIFT), 'no_drift': (list(range(50, 62)), {})}
    report = {}
    for name, (seeds, kw) in sets.items():
        with Pool(2) as pool:
            rows = pool.map(run_seed, [(s, kw) for s in seeds])
        s = summary(rows)
        g = gates_nodrift(s) if name == 'no_drift' else gates_drift(s)
        report[name] = dict(summary=s, gates=g, rows=rows)
        print(name, json.dumps(s), flush=True)
        for k, v in g.items():
            print('  ', k, v, flush=True)
        for r in rows:
            print('    seed', r['seed'], 'drift', r['drift'], 'offset', r['turn_offset'], 'rate', r['turn_rate_ratio'],
                  'merged c/n/a', r['committed']['turn_merged'], r['naive']['turn_merged'], r['aware']['turn_merged'],
                  'naive reopen', [(x['kind'], x['gap'], x['false']) for x in r['reopen_naive']],
                  'aware reopen', [(x['kind'], x['gap'], x['false']) for x in r['reopen_aware']], flush=True)
    out = 'results/drift_calib.json' if calib else 'results/drift_control.json'
    json.dump(report, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
