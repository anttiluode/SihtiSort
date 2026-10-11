"""Run every sorter on a set of seeds and evaluate the gates in PROTOCOL.md.

    python run_gates.py calibration     # seeds 100-102, for debugging only
    python run_gates.py test            # seeds 0-4, run once on frozen code
    python run_gates.py sparse_posthoc  # POST HOC: seeds 10-14, neurons >= 1 channel apart
"""
import json
import subprocess
import sys
import time

import numpy as np

from sihtisort import sort as S
from sihtisort.evaluate import score
from sihtisort.synth import make_recording

SETS = {'calibration': [100, 101, 102], 'test': [0, 1, 2, 3, 4],
        'sparse_posthoc': [10, 11, 12, 13, 14]}       # POST HOC: neurons >= 1 channel apart
MIN_SEP = {'sparse_posthoc': 1.0}
VARIANTS = ['B1', 'B2', 'S', 'S-block', 'S-memory', 'S-sieve']


def run_seed(seed, min_sep=None):
    data, gt, meta = make_recording(seed, min_sep=min_sep)
    base = S.snapshot_sort(data, 0)
    outs = {
        'B1': dict(t=base['t'], unit=base['label'], state=np.zeros(len(base['t']), int)),
        'B2': S.run_b2(data, base=base)[0],
        'S': S.run_s(data, base=base),
        'S-block': S.run_s(data, base=base, block=False),
        'S-memory': S.run_s(data, base=base, memory=False),
        'S-sieve': S.run_s(data, base=base, use_sieve=False),
    }
    res = {k: score(o, gt, meta) for k, o in outs.items()}
    res['S']['ledger'] = outs['S']['ledger']
    res['S']['n_states'] = outs['S']['n_states']
    return res


def pooled(results, v):
    rows = [r for seed in results for r in results[seed][v]['neurons']]
    mood = [r for r in rows if r['kind'].startswith('mood')]
    mild = [r for r in rows if r['kind'] == 'mood_mild']
    strong = [r for r in rows if r['kind'] == 'mood_strong']
    stable = [r for r in rows if r['kind'] == 'stable']
    st_mild = [r['state_acc'] for r in mild if r['one_unit'] and 'state_acc' in r]
    st_strong = [r['state_acc'] for r in strong if r['one_unit'] and 'state_acc' in r]
    return dict(
        mood_one_unit=sum(r['one_unit'] for r in mood), n_mood=len(mood),
        strong_one_unit=sum(r['one_unit'] for r in strong), mild_one_unit=sum(r['one_unit'] for r in mild),
        stable_mean_acc=round(float(np.mean([r['best_acc'] for r in stable])), 4),
        false_merges=sum(results[s][v]['false_merges'] for s in results),
        turn_merged=[(round(results[s][v]['turn_offset'], 2), results[s][v]['turn_merged']) for s in results],
        overlap_recall=round(float(np.mean([results[s][v]['overlap_recall'] for s in results])), 4),
        state_acc_mild=st_mild, state_acc_strong=st_strong)


def gates(P):
    pct = lambda v: round(100.0 * P[v]['mood_one_unit'] / P[v]['n_mood'], 6)
    g = {}
    g['G0'] = dict(B1_mood_pct=pct('B1'), B1_stable_acc=P['B1']['stable_mean_acc'],
                   passed=pct('B1') <= 50 and P['B1']['stable_mean_acc'] >= 0.8)
    g['G1'] = dict(S_mood_pct=pct('S'), B2_mood_pct=pct('B2'), gain=pct('S') - pct('B2'),
                   S_false_merges=P['S']['false_merges'], B2_false_merges=P['B2']['false_merges'],
                   S_stable=P['S']['stable_mean_acc'], B2_stable=P['B2']['stable_mean_acc'],
                   passed=(pct('S') - pct('B2') >= 25 and P['S']['false_merges'] <= P['B2']['false_merges'] + 2
                           and P['S']['stable_mean_acc'] >= P['B2']['stable_mean_acc'] - 0.02))
    g['G2'] = dict(S_block_mood_pct=pct('S-block'), B2_mood_pct=pct('B2'),
                   passed=pct('S-block') - pct('B2') <= 10)
    sm, sn = P['S']['state_acc_mild'], P['S-memory']['state_acc_mild']
    g['G3'] = dict(S_state_mild=sm, S_memory_state_mild=sn,
                   S_state_strong=P['S']['state_acc_strong'], S_memory_state_strong=P['S-memory']['state_acc_strong'],
                   diff_points=(100 * (np.mean(sm) - np.mean(sn)) if sm and sn else None),
                   passed=bool(sm and sn and 100 * (np.mean(sm) - np.mean(sn)) >= 5))
    far = [m for off, m in P['S']['turn_merged'] if off >= 1.25]
    g['G4'] = dict(turn_merged=P['S']['turn_merged'], far_merges=sum(far),
                   merge_rule_failed=sum(far) >= 2)
    g['G5'] = dict(S_overlap=P['S']['overlap_recall'], S_sieve_overlap=P['S-sieve']['overlap_recall'],
                   passed=P['S']['overlap_recall'] >= P['S-sieve']['overlap_recall'])
    return g


def main(which):
    t0 = time.time()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', 'sihtisort'], capture_output=True, text=True).stdout.strip())
    results = {}
    for seed in SETS[which]:
        results[seed] = run_seed(seed, MIN_SEP.get(which))
        print(f'seed {seed} done ({time.time() - t0:.0f} s)', flush=True)
    P = {v: pooled(results, v) for v in VARIANTS}
    G = gates(P)
    out = dict(set=which, seeds=SETS[which], commit=commit, code_dirty=dirty, pooled=P, gates=G,
               per_seed={str(s): results[s] for s in results})
    with open(f'results/{which}.json', 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    for v in VARIANTS:
        p = P[v]
        print(f"{v:9s} mood one-unit {p['mood_one_unit']}/{p['n_mood']} (strong {p['strong_one_unit']}, mild {p['mild_one_unit']})"
              f"  stable acc {p['stable_mean_acc']:.3f}  false merges {p['false_merges']}  overlap {p['overlap_recall']:.3f}")
    for k, v in G.items():
        print(k, v)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'calibration')
