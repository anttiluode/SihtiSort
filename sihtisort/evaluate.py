"""Scoring against ground truth, as defined in PROTOCOL.md."""
import numpy as np

TOL = 8                     # 0.4 ms at 20 kHz
OVERLAP = 20                # 1 ms


def match_idx(gt_t, u_t):
    """For each ground-truth spike, index of a unit spike within TOL, else -1."""
    if len(u_t) == 0:
        return np.full(len(gt_t), -1)
    i = np.clip(np.searchsorted(u_t, gt_t), 1, len(u_t) - 1)
    left, right = u_t[i - 1], u_t[i]
    j = np.where(np.abs(gt_t - left) <= np.abs(gt_t - right), i - 1, i)
    ok = np.abs(u_t[j] - gt_t) <= TOL
    return np.where(ok, j, -1)


def global_match(gt_t, st):
    """One-to-one matching across ALL neurons and units: each true spike and each sorted
    spike is used at most once, closest pairs first. Returns sorted index per true spike, or -1."""
    lo = np.searchsorted(st, gt_t - TOL, 'left'); hi = np.searchsorted(st, gt_t + TOL, 'right')
    cand = [(abs(int(st[j]) - int(gt_t[i])), i, j) for i in range(len(gt_t)) for j in range(lo[i], hi[i])]
    cand.sort()
    out = np.full(len(gt_t), -1); used = np.zeros(len(st), bool)
    for _, i, j in cand:
        if out[i] < 0 and not used[j]:
            out[i] = j; used[j] = True
    return out


def score(sorted_out, gt, meta, one_to_one=True):
    """one_to_one=True is the protocol's rule. one_to_one=False is the original (buggy)
    nearest-spike scoring, kept only to show how much the bug changed the numbers."""
    st, su, ss = sorted_out['t'], sorted_out['unit'], sorted_out['state']
    order = np.argsort(st, kind='stable'); st, su, ss = st[order], su[order], ss[order]
    units = np.unique(su)
    neurons = meta['neurons']
    by_unit = {u: np.where(su == u)[0] for u in units}
    M = np.zeros((len(neurons), len(units)))
    if one_to_one:
        gm = global_match(gt['t'], st)
        kidx = {u: k for k, u in enumerate(units)}
        for g, nu in enumerate(neurons):
            m = (gt['unit'] == nu['id']) & (gm >= 0)
            for j in gm[m]:
                M[g, kidx[su[j]]] += 1
    else:
        for g, nu in enumerate(neurons):
            gt_t = gt['t'][gt['unit'] == nu['id']]
            for k, u in enumerate(units):
                M[g, k] = (match_idx(gt_t, st[by_unit[u]]) >= 0).sum()
    n_g = np.array([(gt['unit'] == nu['id']).sum() for nu in neurons])
    n_u = np.array([len(by_unit[u]) for u in units])
    acc = M / (n_g[:, None] + n_u[None, :] - M)
    best = acc.argmax(1)
    per = []
    for g, nu in enumerate(neurons):
        k = best[g]; u = units[k]
        rec = dict(id=nu['id'], kind=nu['kind'], pos=round(nu['pos'], 2), best_acc=round(float(acc[g, k]), 4),
                   one_unit=bool(acc[g, k] >= 0.8),
                   n_units_10pct=int((M[g] >= 0.1 * n_g[g]).sum()))
        if nu['kind'].startswith('mood'):
            gt_mask = gt['unit'] == nu['id']
            if one_to_one:
                jj = gm[gt_mask]
                ok = (jj >= 0) & (su[np.maximum(jj, 0)] == u)
                dec = ss[jj[ok]]
            else:
                j = match_idx(gt['t'][gt_mask], st[by_unit[u]])
                ok = j >= 0
                dec = ss[by_unit[u]][j[ok]]
            tru = gt['state'][gt_mask][ok]
            n_states = len(np.unique(ss[by_unit[u]]))
            if n_states >= 2 and ok.sum() > 0:
                agree = 0
                for d in np.unique(dec):
                    agree += np.bincount(tru[dec == d], minlength=2).max()
                rec['state_acc'] = round(float(agree / ok.sum()), 4)
        per.append(rec)

    # false merges: two different neurons each >= 20% of a unit's matched spikes
    false_merges, turn_merged = 0, False
    kinds = [nu['kind'] for nu in neurons]
    for k, u in enumerate(units):
        tot = M[:, k].sum()
        if tot == 0:
            continue
        big = np.where(M[:, k] >= 0.2 * tot)[0]
        if len(big) >= 2:
            ks = sorted(kinds[g] for g in big)
            if ks == ['turn_a', 'turn_b']:
                turn_merged = True
            else:
                false_merges += 1

    # overlap recall
    others = []
    hit = 0
    for g, nu in enumerate(neurons):
        gt_t = gt['t'][gt['unit'] == nu['id']]
        oth = gt['t'][gt['unit'] != nu['id']]
        i = np.clip(np.searchsorted(oth, gt_t), 1, len(oth) - 1)
        near = np.minimum(np.abs(gt_t - oth[i - 1]), np.abs(gt_t - oth[i])) <= OVERLAP
        if one_to_one:
            jj = gm[gt['unit'] == nu['id']][near]
            hit += int(((jj >= 0) & (su[np.maximum(jj, 0)] == units[best[g]])).sum())
        else:
            j = match_idx(gt_t[near], st[by_unit[units[best[g]]]])
            hit += (j >= 0).sum()
        others.append(near.sum())
    return dict(neurons=per, false_merges=false_merges, turn_merged=turn_merged,
                turn_offset=meta['turn_offset'], n_units=int(len(units)),
                overlap_recall=round(float(hit / max(1, sum(others))), 4), n_overlap=int(sum(others)))
