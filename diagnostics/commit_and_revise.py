"""PROTOCOL_2: commit, name the refuter, revise.

    python diagnostics/commit_and_revise.py calib    # seeds 40-41, for debugging only
    python diagnostics/commit_and_revise.py          # identifiability set 20-31, then test 0-4 and sparse 10-14
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort import sort as S                          # noqa: E402
from sihtisort.evaluate import global_match, score       # noqa: E402
from sihtisort.synth import make_recording               # noqa: E402

TURN_RANGE = (0.0, 0.5)


def dominant(gm, gt, unit_of_sorted, mask_sorted):
    """True neuron that supplies most matched spikes among the selected sorted spikes."""
    inv = np.full(mask_sorted.shape[0], -1)
    ok = gm >= 0
    inv[gm[ok]] = gt['unit'][ok]
    ids = inv[mask_sorted]; ids = ids[ids >= 0]
    if len(ids) == 0:
        return None, 0.0
    c = np.bincount(ids)
    return int(c.argmax()), float(c.max() / len(ids))


def run(seed, min_sep, turn_range):
    data, gt, meta = make_recording(seed, min_sep=min_sep, turn_range=turn_range)
    base = S.snapshot_sort(data, 0)
    v0_out = S.run_s(data, base=base)                  # SihtiSort v0, unchanged
    out = S.run_s(data, base=base, commit=True)        # PROTOCOL_2 mode
    o = np.argsort(out['t'], kind='stable')
    t, u, s = out['t'][o], out['unit'][o], out['state'][o]
    gm = global_match(gt['t'], t)
    kinds = {n['id']: n['kind'] for n in meta['neurons']}
    rates = {n['id']: n['rate'] for n in meta['neurons']}
    commits = []
    for c in out['commitments']:
        a, b = c['states']
        ga, pa = dominant(gm, gt, u, (u == c['address']) & (s == a))
        gb, pb = dominant(gm, gt, u, (u == c['address']) & (s == b))
        truth = 'one neuron' if ga == gb else 'two neurons'
        c2 = {k: v for k, v in c.items() if k != 'timeline'}
        c2.update(true_a=ga, true_b=gb, kind_a=kinds.get(ga), kind_b=kinds.get(gb), purity=[round(pa, 3), round(pb, 3)],
                  truth=truth, correct=(truth == 'one neuron') == (c['status'] != 'refuted'))
        first_refuted = next((p['t'] for p in c['timeline'] if p['status'] == 'refuted'), None)
        first_survived = next((p['t'] for p in c['timeline'] if p['status'] == 'survived'), None)
        c2.update(first_refuted_s=first_refuted, first_survived_s=first_survived, timeline=c['timeline'])
        commits.append(c2)
    v0 = score(dict(t=v0_out['t'], unit=v0_out['unit'], state=v0_out['state']), gt, meta)
    pre = score(dict(t=out['t'], unit=out['unit'], state=out['state']), gt, meta)
    v1 = score(dict(t=out['t'], unit=out['unit_revised'], state=out['state']), gt, meta)
    ta = [n for n in meta['neurons'] if n['kind'] == 'turn_a'][0]['rate']
    tb = [n for n in meta['neurons'] if n['kind'] == 'turn_b'][0]['rate']
    return dict(seed=seed, turn_offset=meta['turn_offset'], turn_rate_ratio=round(max(ta, tb) / min(ta, tb), 3),
                v0=dict(turn_merged=v0['turn_merged'], mood_one_unit=sum(r['one_unit'] for r in v0['neurons'] if r['kind'].startswith('mood')),
                        false_merges=v0['false_merges']),
                committed_before_revision=dict(turn_merged=pre['turn_merged'], mood_one_unit=sum(r['one_unit'] for r in pre['neurons'] if r['kind'].startswith('mood')),
                                               false_merges=pre['false_merges']),
                revised=dict(turn_merged=v1['turn_merged'], mood_one_unit=sum(r['one_unit'] for r in v1['neurons'] if r['kind'].startswith('mood')),
                             false_merges=v1['false_merges']),
                commitments=commits)


def gates(rows):
    turn_v0 = [r for r in rows if r['v0']['turn_merged'] and r['turn_rate_ratio'] >= 1.5]
    r1_split = sum(not r['revised']['turn_merged'] for r in turn_v0)
    mood_c = [c for r in rows for c in r['commitments'] if c['truth'] == 'one neuron' and str(c['kind_a']).startswith('mood')]
    r2_wrong = sum(c['status'] == 'refuted' for c in mood_c)
    surv = [c for r in rows for c in r['commitments'] if c['status'] == 'survived']
    unt = [c for r in rows for c in r['commitments'] if c['status'] == 'untested from here']
    ref = [c for r in rows for c in r['commitments'] if c['status'] == 'refuted']
    return dict(
        R1=dict(turn_pairs_merged_by_v0_with_rate_ratio_ge_1_5=len(turn_v0), split_by_revision=r1_split,
                passed=bool(turn_v0) and r1_split / len(turn_v0) >= 0.7),
        R1_context=dict(turn_pairs_merged_by_v0_total=sum(r['v0']['turn_merged'] for r in rows),
                        similar_rate_pairs_merged=[(r['seed'], r['turn_rate_ratio'], r['revised']['turn_merged'])
                                                   for r in rows if r['v0']['turn_merged'] and r['turn_rate_ratio'] < 1.5]),
        R2=dict(mood_merges=len(mood_c), wrongly_split=r2_wrong, passed=bool(mood_c) and r2_wrong / len(mood_c) <= 0.10),
        R3=dict(survived=len(surv), survived_correct=sum(c['truth'] == 'one neuron' for c in surv),
                passed=bool(surv) and sum(c['truth'] == 'one neuron' for c in surv) / len(surv) >= 0.9,
                untested=len(unt), untested_truly_one_neuron=sum(c['truth'] == 'one neuron' for c in unt),
                refuted=len(ref), refuted_truly_two_neurons=sum(c['truth'] == 'two neurons' for c in ref)),
        R4=dict(revisions=len(ref), seconds_to_revise=[c['first_refuted_s'] for c in ref],
                seconds_to_survive=[c['first_survived_s'] for c in surv]))


if __name__ == '__main__':
    calib = len(sys.argv) > 1 and sys.argv[1] == 'calib'
    sets = {'calib': ([40, 41], 1.0, TURN_RANGE)} if calib else {
        'identifiability': (list(range(20, 32)), 1.0, TURN_RANGE),
        'test': ([0, 1, 2, 3, 4], None, None),
        'sparse_posthoc': ([10, 11, 12, 13, 14], 1.0, None)}
    report = {}
    for name, (seeds, ms, tr) in sets.items():
        rows = [run(sd, ms, tr) for sd in seeds]
        report[name] = dict(gates=gates(rows),
                            mood_one_unit_v0=sum(r['v0']['mood_one_unit'] for r in rows),
                            mood_one_unit_revised=sum(r['revised']['mood_one_unit'] for r in rows),
                            false_merges_v0=sum(r['v0']['false_merges'] for r in rows),
                            false_merges_revised=sum(r['revised']['false_merges'] for r in rows),
                            rows=rows)
        g = report[name]
        print(name, 'mood one-unit', g['mood_one_unit_v0'], '->', g['mood_one_unit_revised'],
              ' false merges', g['false_merges_v0'], '->', g['false_merges_revised'])
        for k, v in g['gates'].items():
            print('  ', k, v)
        for r in rows:
            for c in r['commitments']:
                print('     seed', r['seed'], c['kind_a'], c['kind_b'], c['truth'], c['status'], 'runs_z', c['confidence_runs_z'],
                      'ratio', c.get('ratio'), 'z', c.get('z'), 'more s', c.get('need_more_seconds'), 'purity', c['purity'])
    out = 'results/commit_revise_calib.json' if calib else 'results/commit_revise.json'
    json.dump(report, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
