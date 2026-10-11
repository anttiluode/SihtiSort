"""PROTOCOL_3: committed address plus remainder, sorted online on growing prefixes.

    python diagnostics/two_channel.py calib     # seeds 70-71, debugging only
    python diagnostics/two_channel.py           # seeds 50-61 (online), then test 0-4 and sparse 10-14 (full only)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort import sort as S                           # noqa: E402
from sihtisort.evaluate import global_match, score        # noqa: E402
from sihtisort.synth import FS, WIN_POST, make_recording  # noqa: E402

PREFIXES = [10, 20, 30, 40, 50, 60]


def counts(sc):
    one = lambda k: sum(r['one_unit'] for r in sc['neurons'] if r['kind'].startswith(k))
    return dict(mood=one('mood'), stable=one('stable'), turn_merged=sc['turn_merged'], false_merges=sc['false_merges'])


def turn_address_remainder(out, gt, meta):
    """Remainder entries for the committed address that holds both turn-taking neurons, if any."""
    o = np.argsort(out['t'], kind='stable'); t, u = out['t'][o], out['unit'][o]
    gm = global_match(gt['t'], t)
    ids = {n['kind']: n['id'] for n in meta['neurons'] if n['kind'].startswith('turn')}
    addr = None
    for a in np.unique(u):
        src = gt['unit'][(gm >= 0)][np.isin(gm[gm >= 0], np.where(u == a)[0])]
        if len(src) and (src == ids['turn_a']).mean() >= 0.2 and (src == ids['turn_b']).mean() >= 0.2:
            addr = int(a)
    return [r for r in out['remainder'] if r['address'] == addr] if addr is not None else []


def one_run(data, gt, meta, seed):
    base = S.snapshot_sort(data, 0)
    out = S.run_s(data, base=base, commit=True, two_channel=True)
    c = score(dict(t=out['t'], unit=out['unit'], state=out['state']), gt, meta)
    tc = score(dict(t=out['t'], unit=out['unit_two_channel'], state=out['state']), gt, meta)
    return out, counts(c), counts(tc)


def run_seed(args):
    seed, min_sep, turn_range, online = args
    data, gt, meta = make_recording(seed, min_sep=min_sep, turn_range=turn_range)
    ta = [n['rate'] for n in meta['neurons'] if n['kind'] == 'turn_a'][0]
    tb = [n['rate'] for n in meta['neurons'] if n['kind'] == 'turn_b'][0]
    rec = dict(seed=seed, turn_offset=round(meta['turn_offset'], 3), turn_rate_ratio=round(max(ta, tb) / min(ta, tb), 3))
    trace = []
    for T in (PREFIXES if online else [60]):
        n = int(T * FS)
        keep = gt['t'] < n - WIN_POST
        g = {k: v[keep] for k, v in gt.items()}
        out, c, tc = one_run(data[:n], g, meta, seed)
        trace.append(dict(T=T, committed=c, two_channel=tc,
                          turn_remainder=turn_address_remainder(out, g, meta),
                          reopened=[r for r in out['remainder'] if r['status'] == 'reopened']))
    rec['trace'] = trace
    rec['final'] = trace[-1]
    return rec


def gates(rows):
    sep = [r for r in rows if r['final']['committed']['turn_merged']
           and (r['turn_offset'] >= 0.25 or r['turn_rate_ratio'] >= 1.5)]
    p1 = sum(not r['final']['two_channel']['turn_merged'] for r in sep)
    mood_c = sum(r['final']['committed']['mood'] for r in rows); mood_t = sum(r['final']['two_channel']['mood'] for r in rows)
    st_c = sum(r['final']['committed']['stable'] for r in rows); st_t = sum(r['final']['two_channel']['stable'] for r in rows)
    G = dict(
        P1=dict(separable_pairs_merged_by_commit_mode=len(sep), reopened_at_60s=p1,
                passed=bool(sep) and p1 / len(sep) >= 0.7),
        P2=dict(mood_one_unit_commit=mood_c, mood_one_unit_two_channel=mood_t, passed=mood_t >= mood_c - 1),
        P3=dict(stable_one_unit_commit=st_c, stable_one_unit_two_channel=st_t, passed=st_t >= st_c - 2),
        false_merges=dict(commit=sum(r['final']['committed']['false_merges'] for r in rows),
                          two_channel=sum(r['final']['two_channel']['false_merges'] for r in rows)))
    if len(rows[0]['trace']) > 1:
        revised, flipped_back = [], []
        for r in rows:
            merged = [p['two_channel']['turn_merged'] for p in r['trace']]
            if not merged[-1] and any(merged[:-1]):
                revised.append(dict(seed=r['seed'], merged_at=[p['T'] for p in r['trace'] if p['two_channel']['turn_merged']],
                                    separated_at=[p['T'] for p in r['trace'] if not p['two_channel']['turn_merged']]))
            if merged[-1] and not all(merged):
                flipped_back.append(r['seed'])
        G['P4'] = dict(online_revisions=revised, separated_then_remerged=flipped_back, passed=not flipped_back)
        G['P5'] = [dict(seed=r['seed'], offset=r['turn_offset'], rate_ratio=r['turn_rate_ratio'],
                        remainder=[(x['status'], x.get('position_gap'), x.get('position_se')) for x in r['final']['turn_remainder']])
                   for r in rows if r['final']['two_channel']['turn_merged']]
    return G


if __name__ == '__main__':
    calib = len(sys.argv) > 1 and sys.argv[1] == 'calib'
    sets = {'calib': ([70, 71], 1.0, (0.0, 0.5), True)} if calib else {
        'identifiability': (list(range(50, 62)), 1.0, (0.0, 0.5), True),
        'test': ([0, 1, 2, 3, 4], None, None, False),
        'sparse_posthoc': ([10, 11, 12, 13, 14], 1.0, None, False)}
    report = {}
    for name, (seeds, ms, tr, online) in sets.items():
        with Pool(2) as pool:
            rows = pool.map(run_seed, [(s, ms, tr, online) for s in seeds])
        report[name] = dict(gates=gates(rows), rows=rows)
        print(name, flush=True)
        for k, v in report[name]['gates'].items():
            print('  ', k, v, flush=True)
        for r in rows:
            print('     seed', r['seed'], 'offset', r['turn_offset'], 'rate', r['turn_rate_ratio'],
                  'turn merged by T (two-channel):', [p['two_channel']['turn_merged'] for p in r['trace']],
                  'reopened:', [(x['kind'], x['by'], x['position_gap']) for x in r['final']['reopened']], flush=True)
    out = 'results/two_channel_calib.json' if calib else 'results/two_channel.json'
    json.dump(report, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
