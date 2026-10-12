"""Snapshot sorters (B1, B2) and SihtiSort (S) with its ablations.

Vocabulary used in the code:
  unit     -- one sorted cluster of spikes (B1, B2)
  address  -- one hypothesised neuron in S; it may hold several waveform states
  ledger   -- every hypothesis S tested, with its numbers and verdict
"""
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .synth import FS, WIN_PRE, WIN_POST

THR = 4.5
DEAD_DET = 10                 # 0.5 ms detection dead time
REFRAC = int(0.0015 * FS)     # 1.5 ms refractory check
FP_MIN = 0.95                 # footprint cosine
CORR_MIN = 0.90               # template correlation (standard merge)
R_MAX = 0.30                  # switch ratio (temporal-context merge)
MIN_N = 30
TAU_MEM = 5.0 * FS            # memory time constant, samples
EXPLAIN = 0.5                 # sieve accepts if removed energy >= EXPLAIN * fitted template energy
SHIFTS = range(-2, 3)
LOOKS = 1                     # rounds of purify -> look again -> regroup
L = WIN_PRE + WIN_POST


# ----------------------------------------------------------------- detection
def noise_sigma(data):
    return np.median(np.abs(data - np.median(data, 0)), 0) / 0.6745


def detect(data, sigma, exclude=None):
    z = (data / sigma).min(1)
    pk, _ = find_peaks(-z, height=THR, distance=DEAD_DET)
    pk = pk[(pk >= WIN_PRE + 2) & (pk < len(data) - WIN_POST - 2)]
    return pk


def snippets(data, times):
    idx = times[:, None] + np.arange(-WIN_PRE, WIN_POST)[None, :]
    return data[idx]                                    # (n, L, C)


# ------------------------------------------------------------ cluster tools
def cluster_bic(X, kmin, kmax, seed=0):
    feats = PCA(n_components=min(10, X.shape[0] - 1, X.shape[1]), random_state=seed).fit_transform(X)
    best, best_bic = None, np.inf
    for k in range(kmin, min(kmax, len(X) // 10) + 1):
        g = GaussianMixture(k, covariance_type='full', n_init=2, random_state=seed, reg_covar=1e-3).fit(feats)
        b = g.bic(feats)
        if b < best_bic:
            best, best_bic = g, b
    return best.predict(feats)


def template_of(snips):
    return np.median(snips, 0)


def fp_of(T):
    f = T.max(0) - T.min(0)
    return f / (np.linalg.norm(f) + 1e-12)


def corr(T1, T2):
    a, b = T1.ravel() - T1.mean(), T2.ravel() - T2.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def viol(times):
    if len(times) < 2:
        return 0.0
    d = np.diff(np.sort(times))
    return float(np.mean(d < REFRAC))


def refractory_ok(t1, t2):
    u = viol(np.concatenate([t1, t2]))
    return u <= max(viol(t1), viol(t2)) + 0.005 and u <= 0.01, u


def switch_ratio(t1, t2):
    t = np.concatenate([t1, t2]); lab = np.r_[np.zeros(len(t1)), np.ones(len(t2))]
    lab = lab[np.argsort(t, kind='stable')]
    n = len(lab); p = len(t1) / n
    exp = 2 * p * (1 - p) * (n - 1)
    return float(np.sum(lab[1:] != lab[:-1]) / max(exp, 1e-9))


# ------------------------------------------------------------- B1 and B2
def snapshot_sort(data, seed=0):
    sigma = noise_sigma(data)
    t = detect(data, sigma)
    S = snippets(data, t)
    lab = cluster_bic(S.reshape(len(S), -1), 4, 20, seed)
    return dict(t=t, label=lab, snips=S, sigma=sigma)


def standard_merge(t, label, snips, ledger=None, turns_to_states=False):
    """Greedy merge of units with the same footprint, correlated templates and a passing refractory check.
    turns_to_states (PROTOCOL_2): parts that take turns are not merged here; they are left for the
    state rule, where the merge becomes a commitment with a refuter."""
    label = label.copy()
    while True:
        ids = np.unique(label)
        T = {i: template_of(snips[label == i]) for i in ids}
        best = None
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                f = float(fp_of(T[i]) @ fp_of(T[j]))
                if f < FP_MIN:
                    continue
                c = corr(T[i], T[j])
                if c < CORR_MIN:
                    continue
                ok, u = refractory_ok(t[label == i], t[label == j])
                verdict = 'accept' if ok else 'reject: refractory'
                if ok and turns_to_states and min((label == i).sum(), (label == j).sum()) >= MIN_N \
                        and switch_ratio(t[label == i], t[label == j]) <= R_MAX:
                    ok, verdict = False, 'defer: same shape but they take turns -> state hypothesis'
                if ledger is not None:
                    ledger.append(dict(test='same neuron, same shape', a=int(i), b=int(j), footprint=round(f, 3),
                                       corr=round(c, 3), union_viol=round(u, 4), verdict=verdict))
                if ok and (best is None or c > best[0]):
                    best = (c, i, j)
        if best is None:
            return label
        label[label == best[2]] = best[1]


# ------------------------------------------------------------ S: addresses
def temporal_merge(t, label, snips, ledger):
    """Group units into addresses: same place, refractory-compatible, taking turns in long blocks."""
    groups = {int(i): [int(i)] for i in np.unique(label)}       # address -> list of member units (states)
    T = {int(i): template_of(snips[label == i]) for i in np.unique(label)}
    times = {int(i): t[label == i] for i in np.unique(label)}

    def addr_fp(a):
        f = sum(fp_of(T[u]) * len(times[u]) for u in groups[a])
        return f / np.linalg.norm(f)

    def addr_t(a):
        return np.concatenate([times[u] for u in groups[a]])

    while True:
        keys = sorted(groups)
        best = None
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                a, b = keys[x], keys[y]
                f = float(addr_fp(a) @ addr_fp(b))
                if f < 0.90:
                    continue
                ta, tb = addr_t(a), addr_t(b)
                rec = dict(test='same neuron, different state', a=a, b=b, footprint=round(f, 3),
                           n=[len(ta), len(tb)])
                if f < FP_MIN:
                    rec['verdict'] = 'reject: different place'
                elif min(len(ta), len(tb)) < MIN_N:
                    rec['verdict'] = 'reject: too few spikes'
                else:
                    ok, u = refractory_ok(ta, tb)
                    R = switch_ratio(ta, tb)
                    rec.update(union_viol=round(u, 4), switch_ratio=round(R, 3))
                    if not ok:
                        rec['verdict'] = 'reject: refractory'
                    elif R > R_MAX:
                        rec['verdict'] = 'reject: they interleave'
                    else:
                        rec['verdict'] = 'accept'
                        if best is None or R < best[0]:
                            best = (R, a, b)
                ledger.append(rec)
        if best is None:
            break
        _, a, b = best
        groups[a] += groups.pop(b)
    return [dict(states=[T[u] for u in groups[a]], members=groups[a]) for a in sorted(groups)]


def addresses_from_units(t, label, snips):
    return [dict(states=[template_of(snips[label == i])], members=[int(i)]) for i in np.unique(label)]


# ---------------------------------------------------------------- the sieve
class Memory:
    """Each address remembers its own state, relaxing to uniform with time since its last spike."""

    def __init__(self, n_states, use):
        self.use = use
        self.belief = [np.ones(k) / k for k in n_states]
        self.last = [None] * len(n_states)

    def logprior(self, a, t):
        k = len(self.belief[a])
        if not self.use or k == 1 or self.last[a] is None:
            return np.zeros(k)
        lam = np.exp(-abs(t - self.last[a]) / TAU_MEM)
        return np.log(np.maximum(lam * self.belief[a] + (1 - lam) / k, 1e-6))

    def update(self, a, t, post):
        self.belief[a] = post
        self.last[a] = t


def fit_all(x, Tm, TT):
    """x: (shifts, D); Tm: (K, D). Returns residual energy (shifts, K) and amplitude."""
    dot = x @ Tm.T
    amp = np.clip(dot / TT, 0.7, 1.3)
    err = (x * x).sum(1)[:, None] - 2 * amp * dot + amp ** 2 * TT
    return err, amp


def sieve(data, sigma, addresses, use_memory=True, passes=3, first_times=None):
    """Matching pursuit on the residue. Returns spikes with address, state and per-state errors."""
    res = data.copy()
    flat, owner = [], []
    for a, ad in enumerate(addresses):
        for s, T in enumerate(ad['states']):
            flat.append(T.ravel()); owner.append((a, s))
    Tm = np.array(flat); TT = (Tm ** 2).sum(1); owner = np.array(owner)
    n_states = [len(ad['states']) for ad in addresses]
    mem = Memory(n_states, use_memory)
    out_t, out_a, out_s, out_err, out_amp, out_k = [], [], [], [], [], []
    last_spike = {}
    unexplained = []
    for p in range(passes):
        ev = first_times if (p == 0 and first_times is not None) else detect(res, sigma)
        if p > 0:
            mem_snapshot = _timeline(out_t, out_a, out_s, len(addresses))
        for t in ev:
            if t - WIN_PRE - 2 < 0 or t + WIN_POST + 2 > len(res):
                continue
            X = np.stack([res[t + sh - WIN_PRE: t + sh + WIN_POST].ravel() for sh in SHIFTS])
            err, amp = fit_all(X, Tm, TT)
            score = -err / 2.0
            prior = np.zeros(len(Tm))
            for a in range(len(addresses)):
                if n_states[a] > 1:
                    lp = mem.logprior(a, t) if p == 0 else _prior_from(mem_snapshot, a, t, n_states[a], use_memory)
                    prior[owner[:, 0] == a] = lp
            score = score + prior[None, :]
            si, k = np.unravel_index(np.argmax(score), score.shape)
            sh = list(SHIFTS)[si]
            e0 = float((X[si] ** 2).sum())
            # accept if the template explains itself: the energy removed is at least half its own energy
            if e0 - err[si, k] < EXPLAIN * amp[si, k] ** 2 * TT[k]:
                unexplained.append(t)
                continue
            a, s = owner[k]
            tt = t + sh
            prev = last_spike.get(a)
            if prev is not None and np.min(np.abs(np.array(prev) - tt)) < REFRAC:
                continue
            res[tt - WIN_PRE: tt + WIN_POST] -= (amp[si, k] * Tm[k]).reshape(L, -1)
            idx = np.where(owner[:, 0] == a)[0]
            e_states = err[si, idx]
            if p == 0 and n_states[a] > 1:
                lp = mem.logprior(a, t)
                ll = -(e_states - e_states.min()) / 2 + lp
                post = np.exp(ll - ll.max()); post /= post.sum()
                mem.update(a, tt, post)
            last_spike.setdefault(a, []).append(tt)
            out_t.append(tt); out_a.append(a); out_s.append(int(s)); out_err.append(e_states)
            out_amp.append(float(amp[si, k])); out_k.append(int(k))
    order = np.argsort(out_t)
    return dict(t=np.array(out_t)[order], addr=np.array(out_a)[order], state=np.array(out_s)[order],
                err=[out_err[i] for i in order], amp=np.array(out_amp)[order], k=np.array(out_k, int)[order],
                Tm=Tm, residue=res, unexplained=np.array(sorted(set(unexplained))))


def _timeline(ts, addrs, states, n_addr):
    ts, addrs, states = np.array(ts), np.array(addrs), np.array(states)
    tl = []
    for a in range(n_addr):
        m = addrs == a
        o = np.argsort(ts[m])
        tl.append((ts[m][o], states[m][o]))
    return tl


def _prior_from(tl, a, t, k, use):
    if not use:
        return np.zeros(k)
    ts, st = tl[a]
    if len(ts) == 0:
        return np.zeros(k)
    i = np.searchsorted(ts, t)
    cand = [j for j in (i - 1, i) if 0 <= j < len(ts)]
    j = min(cand, key=lambda j: abs(ts[j] - t))
    lam = np.exp(-abs(ts[j] - t) / TAU_MEM)
    v = np.full(k, (1 - lam) / k); v[st[j]] += lam
    return np.log(np.maximum(v, 1e-6))


def viterbi_states(t, err, k):
    """Most likely state sequence for one address, emissions -err/2, time-relaxing transitions."""
    n = len(t)
    E = -np.array(err) / 2.0
    V = E[0].copy(); back = np.zeros((n, k), int)
    for i in range(1, n):
        lam = np.exp(-(t[i] - t[i - 1]) / TAU_MEM)
        P = np.log(np.maximum(lam * np.eye(k) + (1 - lam) / k, 1e-12))
        M = V[:, None] + P
        back[i] = M.argmax(0)
        V = M.max(0) + E[i]
    s = np.zeros(n, int); s[-1] = V.argmax()
    for i in range(n - 1, 0, -1):
        s[i - 1] = back[i, s[i]]
    return s


def finalize_states(res_sieve, addresses, use_memory):
    st = res_sieve['state'].copy()
    for a, ad in enumerate(addresses):
        k = len(ad['states'])
        m = np.where(res_sieve['addr'] == a)[0]
        if k < 2 or len(m) == 0:
            continue
        errs = [res_sieve['err'][i] for i in m]
        if use_memory:
            st[m] = viterbi_states(res_sieve['t'][m], errs, k)
        else:
            st[m] = np.argmin(np.array(errs), 1)
    return st


def open_addresses(data, res_sieve, addresses, ledger, seed=0):
    """Cluster what nothing explained; open an address for each cluster that behaves like a neuron."""
    u = res_sieve['unexplained']
    u = u[(u >= WIN_PRE + 2) & (u < len(data) - WIN_POST - 2)]
    if len(u) < 60:
        ledger.append(dict(test='open new address from residue', n_unexplained=int(len(u)), verdict='skip: too few'))
        return addresses
    S = snippets(res_sieve['residue'], u)
    lab = cluster_bic(S.reshape(len(S), -1), 1, 10, seed)
    new = list(addresses)
    for c in np.unique(lab):
        m = lab == c
        T = template_of(S[m]); v = viol(u[m]); depth = float(-T.min())
        ok = m.sum() >= MIN_N and v <= 0.01 and depth >= THR
        ledger.append(dict(test='open new address from residue', n=int(m.sum()), viol=round(v, 4),
                           depth=round(depth, 2), verdict='accept' if ok else 'reject'))
        if ok:
            new.append(dict(states=[T], members=['residue']))
    return new


# ------------------------------------------------------------------ drivers
def run_b1(data, seed=0):
    r = snapshot_sort(data, seed)
    return dict(t=r['t'], unit=r['label'], state=np.zeros(len(r['t']), int)), r


def run_b2(data, seed=0, base=None):
    r = base or snapshot_sort(data, seed)
    lab = standard_merge(r['t'], r['label'], r['snips'])
    return dict(t=r['t'], unit=lab, state=np.zeros(len(r['t']), int)), r


def group(t, lab, snips, ledger, block, commit=False):
    """Standard merges, then (if block) the temporal-context merge into multi-state addresses."""
    lab = standard_merge(t, lab, snips, ledger, turns_to_states=commit)
    if block:
        return lab, temporal_merge(t, lab, snips, ledger)
    return lab, addresses_from_units(t, lab, snips)


def look_again(sv, seed, ledger):
    """Purify every assigned spike (its window of residue plus its own fitted template),
    then split each address on the purified snippets: every part is a new hypothesis."""
    t = sv['t']
    clean = snippets(sv['residue'], t) + sv['amp'][:, None, None] * sv['Tm'][sv['k']].reshape(len(t), L, -1)
    lab = np.zeros(len(t), int); nxt = 0
    for a in np.unique(sv['addr']):
        m = np.where(sv['addr'] == a)[0]
        if len(m) < 60:
            lab[m] = nxt; nxt += 1
            continue
        part = cluster_bic(clean[m].reshape(len(m), -1), 1, 3, seed)
        k = len(np.unique(part))
        ledger.append(dict(test='look again inside address', address=int(a), n=int(len(m)), parts=int(k)))
        lab[m] = nxt + part; nxt += k
    _, lab = np.unique(lab, return_inverse=True)
    return t, lab, clean


def run_s(data, seed=0, base=None, block=True, memory=True, use_sieve=True, commit=False, two_channel=False):
    """commit=True: PROTOCOL_2 mode (turn-taking parts become state hypotheses with refuters)."""
    r = base or snapshot_sort(data, seed)
    ledger = []
    lab, addresses = group(r['t'], r['label'], r['snips'], ledger, block, commit)
    if not use_sieve:
        # merges and memory on the original detections, no residue
        unit_of = {}
        for a, ad in enumerate(addresses):
            for s, mbr in enumerate(ad['members']):
                unit_of[mbr] = (a, s)
        A = np.array([unit_of[int(l)][0] for l in lab]); St = np.array([unit_of[int(l)][1] for l in lab])
        flat = r['snips'].reshape(len(r['t']), -1)
        errs = []
        for i in range(len(r['t'])):
            Ts = np.array([T.ravel() for T in addresses[A[i]]['states']])
            e, _ = fit_all(flat[i:i + 1], Ts, (Ts ** 2).sum(1))
            errs.append(e[0])
        fake = dict(t=r['t'], addr=A, state=St, err=errs)
        st = finalize_states(fake, addresses, memory)
        return dict(t=r['t'], unit=A, state=st, ledger=ledger, n_states=[len(a['states']) for a in addresses])
    sv = sieve(data, r['sigma'], addresses, memory, first_times=r['t'])
    for _ in range(LOOKS):
        t2, lab2, clean = look_again(sv, seed, ledger)
        lab2, addresses = group(t2, lab2, clean, ledger, block, commit)
        sv = sieve(data, r['sigma'], addresses, memory, first_times=r['t'])
    addresses = open_addresses(data, sv, addresses, ledger, seed)
    sv = sieve(data, r['sigma'], addresses, memory, first_times=r['t'])
    st = finalize_states(sv, addresses, memory)
    commits = commitments(sv['t'], sv['addr'], st)
    unit_rev = revise(sv['addr'], st, commits)
    out = dict(t=sv['t'], unit=sv['addr'], state=st, ledger=ledger,
               n_states=[len(a['states']) for a in addresses],
               unit_revised=unit_rev, commitments=commits)
    if two_channel:
        clean = snippets(sv['residue'], sv['t']) + sv['amp'][:, None, None] * sv['Tm'][sv['k']].reshape(len(sv['t']), L, -1)
        out['unit_two_channel'], out['remainder'] = remainder_channel(sv['t'], sv['addr'], st, clean, seed)
        out['unit_two_channel_aware'], out['remainder_aware'] = remainder_channel(sv['t'], sv['addr'], st, clean, seed,
                                                                                  aware=True)
        out['_rem_inputs'] = (sv['t'], sv['addr'], st, clean)      # in memory only (PROTOCOL_5 methods)
    return out


# ------------------------------------------------ PROTOCOL_3: the remainder
POS_MIN = 0.15            # channels
POS_Z = 3.0
POS_SE_TESTED = 0.05      # the position test had power to see a 0.15 gap


def spike_centroid(snips):
    f = snips.max(1) - snips.min(1)
    f = np.clip(f - 0.2 * f.max(1, keepdims=True), 0, None)
    return (f * np.arange(f.shape[1])).sum(1) / np.clip(f.sum(1), 1e-9, None)


def template_centroid(T):
    f = T.max(0) - T.min(0)
    f = np.clip(f - 0.2 * f.max(), 0, None)
    return float((f * np.arange(len(f))).sum() / max(f.sum(), 1e-9))


def template_gap(Sa, Sb, seed=0, boots=40):
    """Position gap between the mean spikes of two parts, with a bootstrap standard error.
    (Per-spike centroids are biased toward the probe centre for small spikes; averaged templates are not.)"""
    gap = abs(template_centroid(Sa.mean(0)) - template_centroid(Sb.mean(0)))
    rng = np.random.default_rng(seed)
    bs = [abs(template_centroid(Sa[rng.integers(0, len(Sa), len(Sa))].mean(0))
              - template_centroid(Sb[rng.integers(0, len(Sb), len(Sb))].mean(0))) for _ in range(boots)]
    return float(gap), float(np.std(bs))


WIN = 15 * FS
MIN_WIN = 6


def window_gap(tp, Sp, tq, Sq, seed=0, boots=40):
    """Drift-aware position gap: compare the two parts only within shared 15 s windows.
    Returns (|weighted mean signed gap|, two-stage bootstrap SE), or (None, None) if no window is shared."""
    lo = min(tp.min(), tq.min())
    wins = []
    for a in np.arange(lo, max(tp.max(), tq.max()) + 1, WIN):
        mp = (tp >= a) & (tp < a + WIN); mq = (tq >= a) & (tq < a + WIN)
        if mp.sum() >= MIN_WIN and mq.sum() >= MIN_WIN:
            wins.append((Sp[mp], Sq[mq]))
    if not wins:
        return None, None

    def est(ws):
        g = np.array([template_centroid(a.mean(0)) - template_centroid(b.mean(0)) for a, b in ws])
        w = np.array([min(len(a), len(b)) for a, b in ws], float)
        return float((g * w).sum() / w.sum())

    gap = abs(est(wins))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boots):
        pick = rng.integers(0, len(wins), len(wins))
        bs.append(est([(wins[i][0][rng.integers(0, len(wins[i][0]), len(wins[i][0]))],
                        wins[i][1][rng.integers(0, len(wins[i][1]), len(wins[i][1]))]) for i in pick]))
    return gap, float(np.std(bs))


SWITCH_K = 15


def general_gap(tp, Sp, tq, Sq, mode, drift_fn=None, seed=0, boots=40, register=False, info=None):
    """PROTOCOL_5 position gap. Groups of (part p spikes, part q spikes) compared at matched conditions:
    'naive' one group (whole parts); 'window' shared 15 s windows; 'switch' the <= 15 spikes on each side
    of every label switch in the time-ordered union. Positions are corrected by drift_fn if given.
    PROTOCOL_6: register=True shifts every spike by -drift at its own time before averaging, instead of
    correcting the averaged centroid. info (a dict) receives the time span (s) of every group."""
    groups = []
    if mode == 'naive':
        groups = [(tp, Sp, tq, Sq)]
    elif mode == 'window':
        lo = min(tp.min(), tq.min())
        for a in np.arange(lo, max(tp.max(), tq.max()) + 1, WIN):
            mp = (tp >= a) & (tp < a + WIN); mq = (tq >= a) & (tq < a + WIN)
            if mp.sum() >= MIN_WIN and mq.sum() >= MIN_WIN:
                groups.append((tp[mp], Sp[mp], tq[mq], Sq[mq]))
    elif mode == 'switch':
        tt = np.concatenate([tp, tq]); lab = np.r_[np.zeros(len(tp), int), np.ones(len(tq), int)]
        idx = np.r_[np.arange(len(tp)), np.arange(len(tq))]
        o = np.argsort(tt, kind='stable'); lab, idx = lab[o], idx[o]
        starts = np.r_[0, np.where(lab[1:] != lab[:-1])[0] + 1]; ends = np.r_[starts[1:], len(lab)]
        for r in range(len(starts) - 1):
            a0, a1, b0, b1 = starts[r], ends[r], starts[r + 1], ends[r + 1]
            before = idx[max(a0, a1 - SWITCH_K):a1]; after = idx[b0:min(b1, b0 + SWITCH_K)]
            if len(before) < MIN_WIN or len(after) < MIN_WIN:
                continue
            if lab[a0] == 0:
                groups.append((tp[before], Sp[before], tq[after], Sq[after]))
            else:
                groups.append((tp[after], Sp[after], tq[before], Sq[before]))
    if info is not None:
        info['spans_s'] = [float((max(ta.max(), tb.max()) - min(ta.min(), tb.min())) / FS) for ta, _, tb, _ in groups]
    if not groups:
        return None, None
    if register and drift_fn is not None:
        groups = [(ta, register_spikes(a, drift_fn(ta)), tb, register_spikes(b, drift_fn(tb))) for ta, a, tb, b in groups]
        drift_fn = None
    d = drift_fn if drift_fn is not None else (lambda x: np.zeros(len(x)))

    def est(gs):
        g = np.array([(template_centroid(a.mean(0)) - float(np.mean(d(ta))))
                      - (template_centroid(b.mean(0)) - float(np.mean(d(tb)))) for ta, a, tb, b in gs])
        w = np.array([min(len(a), len(b)) for _, a, _, b in gs], float)
        return float((g * w).sum() / w.sum())

    gap = abs(est(groups))
    if boots == 0:
        return gap, float('nan')
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boots):
        pick = rng.integers(0, len(groups), len(groups)) if len(groups) > 1 else [0]
        res = []
        for i in pick:
            ta, a, tb, b = groups[i]
            ia = rng.integers(0, len(a), len(a)); ib = rng.integers(0, len(b), len(b))
            res.append((ta[ia], a[ia], tb[ib], b[ib]))
        bs.append(est(res))
    return gap, float(np.std(bs))


def register_spikes(S, shifts):
    """Move each spike's footprint by -shift channels (linear interpolation along the channel axis,
    edge channels repeated). S: (n, L, C); shifts: (n,) in channels."""
    n, Lw, C = S.shape
    x = np.arange(C)[None, :] + np.asarray(shifts, float)[:, None]          # read position for each new channel
    x = np.clip(x, 0, C - 1)
    i0 = np.floor(x).astype(int); i1 = np.minimum(i0 + 1, C - 1); f = (x - i0)[:, None, :]
    g0 = np.take_along_axis(S, np.repeat(i0[:, None, :], Lw, 1), 2)
    g1 = np.take_along_axis(S, np.repeat(i1[:, None, :], Lw, 1), 2)
    return (1 - f) * g0 + f * g1


def estimate_drift(t, addr, clean, win=5 * FS, min_spikes=6, min_windows=3):
    """PROTOCOL_5: blind estimate of the shared probe displacement from all committed addresses.
    Returns a function samples -> channels (zero median), or None if no address qualifies."""
    T = float(t.max()) if len(t) else 0.0
    edges = np.arange(0, T + win, win); centers = edges[:-1] + win / 2
    rows = []
    for a in np.unique(addr):
        m = addr == a
        tr = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            k = m & (t >= edges[i]) & (t < edges[i + 1])
            if k.sum() >= min_spikes:
                tr[i] = template_centroid(clean[k].mean(0))
        if np.isfinite(tr).sum() >= min_windows:
            rows.append(tr - np.nanmedian(tr))
    if not rows:
        return None
    dhat = np.nanmedian(np.array(rows), 0)
    ok = np.isfinite(dhat)
    if ok.sum() < 2:
        return None
    xc, yc = centers[ok], dhat[ok]
    return lambda x: np.interp(np.asarray(x, float), xc, yc)


NULL_DRAWS = 30


def forced_split_null(t, S, null_drift, test_drift, register, seed=0, draws=NULL_DRAWS):
    """PROTOCOL_7: what the forced-split procedure produces from ONE neuron like this address.
    The address's own registered template is put back at each spike's time and drifted position, scaled by
    the spike's own amplitude, plus the address's own residuals in a random order; the identical split and gap
    measurement is then run `draws` times. (Deviation found on calibration seeds: resampling residuals WITH
    replacement duplicates outlier residuals, and the mixture then splits off those duplicates as a small
    cluster, so the null measured the wrong thing. Each residual is now used exactly once per draw.)"""
    n = len(t)
    d = null_drift(t) if null_drift is not None else np.zeros(n)
    Sr = register_spikes(S, d) if np.any(d) else S
    Tbar = Sr.mean(0)
    Tk = register_spikes(np.repeat(Tbar[None], n, 0), -np.asarray(d)) if np.any(d) else np.repeat(Tbar[None], n, 0)
    amp = (S * Tk).sum((1, 2)) / np.maximum((Tk * Tk).sum((1, 2)), 1e-9)
    resid = S - amp[:, None, None] * Tk
    rng = np.random.default_rng(seed + 7)
    gaps, rz = [], []
    for k in range(draws):
        Sn = amp[:, None, None] * Tk + resid[rng.permutation(n)]
        lab, test = _forced_split(Sn.reshape(n, -1), seed + 1 + k)
        p, q = np.where(lab == 0)[0], np.where(lab == 1)[0]
        tp, tq = np.where((lab == 0) & test)[0], np.where((lab == 1) & test)[0]
        if min(len(tp), len(tq)) < MIN_N // 2:
            gaps.append(0.0); rz.append(0.0); continue
        if test_drift is None:
            g, _ = general_gap(t[tp], Sn[tp], t[tq], Sn[tq], 'naive', None, seed, boots=0)
        else:
            g, _ = general_gap(t[tp], Sn[tp], t[tq], Sn[tq], 'naive', test_drift, seed, boots=0, register=register)
        gaps.append(g)
        z = 0.0
        if switch_ratio(t[p], t[q]) <= R_MAX:
            tt = np.concatenate([t[p], t[q]]); ll = np.r_[np.zeros(len(p), int), np.ones(len(q), int)]
            o = np.argsort(tt, kind='stable'); z = refute(tt[o], ll[o], 0, 1).get('z') or 0.0
        rz.append(float(z))
    return dict(gap_mu=round(float(np.mean(gaps)), 4), gap_sd=round(float(np.std(gaps)), 4),
                rate_z_mu=round(float(np.mean(rz)), 3), rate_z_sd=round(float(np.std(rz)), 3), draws=draws)


def _forced_split(X, seed):
    """Two-way split fitted on even-indexed spikes only; the test then uses odd-indexed spikes,
    so the split cannot manufacture the difference it is tested on."""
    fit = np.arange(len(X)) % 2 == 0
    pca = PCA(n_components=5, random_state=seed).fit(X[fit])
    g = GaussianMixture(2, covariance_type="full", n_init=5, random_state=seed, reg_covar=1e-3).fit(pca.transform(X[fit]))
    return g.predict(pca.transform(X)), ~fit


def remainder_channel(t, addr, state, clean, seed=0, aware=False, mode=None, drift_fn=None, register=False,
                      calibrate=False, null_drift=None):
    """aware=True (PROTOCOL_4): compare the parts' positions only inside shared 15 s windows.
    PROTOCOL_5: mode in {'naive', 'window', 'switch'}; drift_fn(samples) -> probe displacement in
    channels, subtracted from positions before comparing (None = no correction).
    PROTOCOL_7: calibrate=True judges forced splits against a one-neuron null built from the address's
    own template, residuals and the measured drift null_drift (which may differ from drift_fn)."""
    if mode is None:
        mode = 'window' if aware else 'naive'
    if mode != 'naive' or drift_fn is not None:
        aware = None      # handled by the general path below
    unit = addr.copy(); nxt = int(addr.max()) + 1 if len(addr) else 0
    rem = []
    for a in np.unique(addr):
        m = np.where(addr == a)[0]
        sts = np.unique(state[m])
        if len(sts) >= 2:
            counts = {int(s): int((state[m] == s).sum()) for s in sts}
            ref = max(counts, key=counts.get)
            parts = [(state[m] == ref, state[m] == s, 'states') for s in sts if s != ref]
            test_mask = np.ones(len(m), bool)
        elif len(m) >= 60:
            lab, test_mask = _forced_split(clean[m].reshape(len(m), -1), seed)
            parts = [(lab == 0, lab == 1, 'forced split')]
        else:
            rem.append(dict(address=int(a), n=int(len(m)), status='silent', reason='too few spikes'))
            continue
        for pm, qm, kind in parts:
            p, q = m[pm], m[qm]
            tp, tq = m[pm & test_mask], m[qm & test_mask]
            rec = dict(address=int(a), kind=kind, n=[int(len(p)), int(len(q))])
            if min(len(tp), len(tq)) < MIN_N // 2:
                rec.update(status='silent', reason='a part is too small')
                rem.append(rec); continue
            if aware is None:
                ginfo = {}
                gap, se = general_gap(t[tp], clean[tp], t[tq], clean[tq], mode, drift_fn, seed, register=register, info=ginfo)
                if mode == 'switch':
                    rec_spans = ginfo.get('spans_s', [])
                if gap is None:
                    gap, se = 0.0, np.inf
            elif aware:
                gap, se = window_gap(t[tp], clean[tp], t[tq], clean[tq], seed)
                if gap is None:
                    gap, se = 0.0, np.inf
            else:
                gap, se = template_gap(clean[tp], clean[tq], seed)
            zp = gap / max(se, 1e-9)
            R = switch_ratio(t[p], t[q])
            rate = None
            if R <= R_MAX:
                tt = np.concatenate([t[p], t[q]]); ll = np.r_[np.zeros(len(p), int), np.ones(len(q), int)]
                o = np.argsort(tt, kind='stable'); rate = refute(tt[o], ll[o], 0, 1)
            pos_fire = gap >= POS_MIN and zp >= POS_Z
            rate_fire = rate is not None and rate['status'] == 'refuted'
            if calibrate and kind == 'forced split':
                nl = forced_split_null(t[m], clean[m], null_drift, drift_fn, register, seed)
                bar = max(POS_MIN, nl['gap_mu'] + 3 * nl['gap_sd'])
                rbar = nl['rate_z_mu'] + 3 * nl['rate_z_sd']
                pos_fire = pos_fire and gap >= bar
                rate_fire = rate_fire and (rate.get('z') or 0) >= rbar
                rec['null'] = dict(nl, position_bar=round(bar, 3), rate_z_bar=round(rbar, 2))
            if aware is None and mode == 'switch':
                rec['switch_spans_s'] = [round(x, 2) for x in rec_spans]
            rec.update(position_gap=round(gap, 3), position_se=round(se, 3) if np.isfinite(se) else None, position_z=round(zp, 2),
                       switch_ratio=round(R, 3), rate=rate)
            if pos_fire or rate_fire:
                rec['status'] = 'reopened'
                rec['by'] = [w for w, f in (('position', pos_fire), ('rate', rate_fire)) if f]
                rec['new_unit'] = nxt
                unit[q] = nxt; nxt += 1
            elif se <= POS_SE_TESTED:
                rec['status'] = 'no contradiction at tested resolution'
            else:
                rec['status'] = 'silent'
            rem.append(rec)
    return unit, rem


# ------------------------------------------------- commit, refute, revise
# PROTOCOL_2.md: each "one neuron, two states" merge is a commitment with a confidence,
# a named refuter (one neuron keeps its rate across states) and a status.
REFUTE_Z = 3.0
REFUTE_RATIO = 1.5
SE_TESTED = 0.135
STEP = 5 * FS


def runs_z(lab):
    """Wald-Wolfowitz runs test on a two-label sequence; negative = fewer runs = taking turns."""
    lab = np.asarray(lab); N = len(lab); n1 = int((lab == lab[0]).sum()) if N else 0; n2 = N - n1
    if N < 3 or n1 == 0 or n2 == 0:
        return 0.0
    runs = 1 + int((lab[1:] != lab[:-1]).sum())
    mu = 2.0 * n1 * n2 / N + 1
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - N) / (N ** 2 * (N - 1))
    return float((runs - mu) / np.sqrt(var)) if var > 0 else 0.0


def state_rate(t, lab, s):
    """Interval count and summed duration over runs of state s with >= 2 spikes."""
    k, dur, i, n = 0, 0.0, 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        if lab[i] == s and j > i:
            k += j - i; dur += float(t[j] - t[i])
        i = j + 1
    return k, dur


def refute(t, lab, a, b):
    ka, da = state_rate(t, lab, a); kb, db = state_rate(t, lab, b)
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    if min(ka, kb) < 1 or min(da, db) <= 0:
        return dict(status='untested from here', reason='a state has no run of two spikes yet')
    ra, rb = ka / da * FS, kb / db * FS
    se = float(np.sqrt(1 / ka + 1 / kb)); z = abs(np.log(ra / rb)) / se
    ratio = max(ra, rb) / min(ra, rb)
    out = dict(rate_a=round(ra, 2), rate_b=round(rb, 2), ratio=round(ratio, 3), z=round(float(z), 2), se=round(se, 3))
    if z >= REFUTE_Z and ratio >= REFUTE_RATIO:
        out['status'] = 'refuted'
    elif se <= SE_TESTED:
        out['status'] = 'survived'
    else:
        f = (se / SE_TESTED) ** 2
        out.update(status='untested from here', need_more_seconds=round((f - 1) * span / FS, 1))
    return out


def commitments(t, unit, state):
    o = np.argsort(t, kind='stable'); t, unit, state = t[o], unit[o], state[o]
    out = []
    for u in np.unique(unit):
        m = unit == u
        ss = np.unique(state[m])
        if len(ss) < 2:
            continue
        counts = {int(s): int((state[m] == s).sum()) for s in ss}
        ref = max(counts, key=counts.get)
        for s in ss:
            if s == ref:
                continue
            mm = m & ((state == ref) | (state == s))
            tt, ll = t[mm], state[mm]
            rec = dict(address=int(u), states=[int(ref), int(s)], n=[counts[ref], counts[int(s)]],
                       confidence_runs_z=round(runs_z(ll), 2), **refute(tt, ll, ref, s))
            timeline = []
            for T in np.arange(tt[0] + STEP, tt[-1] + STEP, STEP):
                k = tt <= T
                if k.sum() < 3:
                    continue
                r = refute(tt[k], ll[k], ref, s)
                timeline.append(dict(t=round(float(T) / FS, 1), runs_z=round(runs_z(ll[k]), 2), status=r['status'],
                                     z=r.get('z'), ratio=r.get('ratio')))
            rec['timeline'] = timeline
            out.append(rec)
    return out


def revise(unit, state, commits):
    """Split every refuted state off into its own address."""
    unit = unit.copy(); nxt = int(unit.max()) + 1 if len(unit) else 0
    for c in commits:
        if c['status'] == 'refuted':
            m = (unit == c['address']) & (state == c['states'][1])
            unit[m] = nxt; nxt += 1
    return unit
