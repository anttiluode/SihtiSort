"""Protocol R2: probe the R1 neuron with small random-sign currents and test the two-site gain
invariant (PROTOCOL_R2.md).

If every bit of state dependence between two sites and the recording is one pointwise gain applied
after their signals merge (y = f(v), v linear), then for every state bin b

    K1(tau|b) / K1(tau)  =  K2(tau|b) / K2(tau),

so  Delta_b(tau) = K1(tau|b)/K1(tau) - K2(tau|b)/K2(tau)  is zero in expectation.
"""
import numpy as np
from scipy.signal import lfilter

from . import cable as K

LAGS = 75              # observation steps (2 ms): 0-148 ms
N_BINS = 4
WINDOW_FRAC = 0.25     # lags where both pooled kernels exceed this fraction of their own peak
N_PERM = 200
T1F_TAU = 8.0          # ms, the low-pass after the tanh in T1f
EPS = 1.0              # probe current (mV/ms at C=1); set on calibration seeds 920-922
DURATION = 300_000.0   # ms of regime A per recording
STATE = 'branch'       # which observable labels the state (see analyse); set on calibration seeds


def make_probed_cell(seed, eps=EPS, duration=DURATION, p=K.PARAMS, clean=True, regime='A'):
    """R1's cell for this seed (same morphology and synapses as cable.make_cell), recorded in
    for `duration` ms and probed at every site. Regime 'A': independent Poisson input at every site.
    Regime 'B': R1's branch bursts (one branch at a time, both its sites) over Poisson at half rate.

    Returns observations for T0, T1, T1f, T2, noisy and (optionally) noise-free, and the probe signs."""
    rng = np.random.default_rng(seed)
    morph = K.morphology(rng, p)
    n_sites = len(morph['sites'])
    w_ampa = p['w_ampa'] * rng.uniform(0.6, 1.4, n_sites)
    w_nmda = p['w_nmda'] * rng.uniform(0.6, 1.4, n_sites)
    steps = int(duration / K.DT)
    rate = p['rate_a'] if regime == 'A' else p['rate_a'] / 2.0
    counts = rng.poisson(rate * K.DT / 1000.0, (steps, n_sites)).astype(float)
    if regime == 'B':
        brng = np.random.default_rng(seed + 9_000_000)
        tb = brng.exponential(p['burst_every'])
        while tb < duration - p['burst_window']:
            br = int(brng.integers(K.N_BRANCH))
            for site in range(br * K.SITES_PER_BRANCH, (br + 1) * K.SITES_PER_BRANCH):
                ts = tb + brng.uniform(0, p['burst_window'], p['burst_spikes'])
                np.add.at(counts[:, site], (ts / K.DT).astype(int), 1.0)
            tb += p['burst_window'] + brng.exponential(p['burst_every'])
    a = np.exp(-K.DT / p['ou_tau'])
    ou = lfilter([1.0], [1.0, -a], rng.standard_normal((steps, morph['n'])) * (p['ou_sd'] * np.sqrt(1 - a * a)), axis=0)
    meas = rng.standard_normal(steps // K.OBS) * p['meas_sd']
    # T1's saturation scale as in R1 (2x the noise-free passive soma sd), from the first 60 s unprobed
    n60 = min(steps, int(60_000.0 / K.DT))
    c0 = K.simulate(morph, counts[:n60], w_ampa, w_nmda, np.zeros((n60, morph['n'])), nmda=False)[0][::K.OBS]
    s = p['tanh_rel'] * float(np.std(c0))

    prng = np.random.default_rng(seed + 7_000_000)
    n_obs = steps // K.OBS
    xi = prng.choice([-1.0, 1.0], size=(n_obs, n_sites))
    xi_ref = prng.choice([-1.0, 1.0], size=(n_obs, n_sites))     # never injected: noise reference
    cur = np.zeros((steps, morph['n']))
    cur[:, morph['sites']] = np.repeat(xi, K.OBS, axis=0) * eps

    af = np.exp(-K.DT / T1F_TAU)
    out = {}

    def run(label, m):
        v0 = K.simulate(morph, counts, w_ampa, w_nmda, cur, nmda=False)[0]
        v2 = K.simulate(morph, counts, w_ampa, w_nmda, cur, nmda=True)[0]
        sat = s * np.tanh((v0 - K.E_L) / s)
        filt = lfilter([1.0 - af], [1.0, -af], sat)
        out[label] = {'T0': v0[::K.OBS] + m, 'T1': K.E_L + sat[::K.OBS] + m,
                      'T1f': K.E_L + filt[::K.OBS] + m, 'T2': v2[::K.OBS] + m}

    if clean:
        run('clean', 0.0)          # probes only: no OU current, no measurement noise
    cur += ou                      # in place, to keep memory down on long recordings
    del ou
    run('noisy', meas)
    return dict(y=out, xi=xi, xi_ref=xi_ref, site_branch=morph['site_branch'], s=s, eps=eps, counts=counts)


# ---------------------------------------------------------------- kernels and the statistic
def _design(y, a, b):
    """Probe times t in [a, b - LAGS - 1) and the lagged responses y[t+1 : t+1+LAGS]."""
    t = np.arange(a, b - LAGS - 1)
    Y = np.lib.stride_tricks.sliding_window_view(y, LAGS)[t + 1]
    return t, Y


def bin_labels(y, t):
    q = np.quantile(y[t], np.linspace(0, 1, N_BINS + 1)[1:-1])
    return np.searchsorted(q, y[t])


def binned_kernels(Y, xi_t, labels, eps):
    """(N_BINS, sites, LAGS) per-bin kernels and (sites, LAGS) pooled, by reverse correlation."""
    T, S = xi_t.shape
    onehot = np.zeros((T, N_BINS), dtype=Y.dtype)
    onehot[np.arange(T), labels] = 1.0
    W = (onehot[:, :, None] * xi_t.astype(Y.dtype)[:, None, :]).reshape(T, N_BINS * S)
    sums = (W.T @ Y).astype(float).reshape(N_BINS, S, -1)
    n = onehot.sum(0).astype(float)
    Kb = sums / (n[:, None, None] * eps)
    Kp = (xi_t.T @ Y) / (T * eps)
    return Kb, Kp


def pair_windows(Kp):
    S = Kp.shape[0]
    pairs, wins = [], []
    for i in range(S):
        for j in range(i + 1, S):
            w = (Kp[i] >= WINDOW_FRAC * Kp[i].max()) & (Kp[j] >= WINDOW_FRAC * Kp[j].max())
            pairs.append((i, j))
            wins.append(w)
    return pairs, wins


def statistic(Kb, Kp, pairs, wins, select=None, key=None):
    """Mean over pairs of mean_{bins, window lags} Delta_b(tau)^2. Kb is one (bins, sites, lags)
    array, or a dict of them keyed by label set with key(i, j) choosing the set for a pair."""
    vals = []
    for (i, j), w in zip(pairs, wins):
        if select is not None and not select(i, j):
            continue
        if w.sum() == 0:
            continue
        kb = Kb if key is None else Kb[key(i, j)]
        d = kb[:, i, w] / Kp[i, w] - kb[:, j, w] / Kp[j, w]
        vals.append(np.mean(d ** 2))
    return float(np.mean(vals)) if vals else float('nan')


def decay_time(k):
    """Lags (ms) from the kernel's peak until it first falls below 1/e of the peak."""
    p = int(np.argmax(k))
    below = np.nonzero(k[p:] < k[p] / np.e)[0]
    return float((below[0] if len(below) else len(k) - p) * K.DT_OBS)


def _probe_prediction(xi, Kp, eps, n):
    """Pooled probe responses: each site's sign train convolved with its pooled kernel (one step late)."""
    out = np.zeros(n)
    for s in range(xi.shape[1]):
        out += np.convolve(xi[:n, s] * eps, np.concatenate([[0.0], Kp[s]]))[:n]
    return out


def analyse(y, xi, site_branch, eps, X, xi_ref=None, n_perm=N_PERM, seed=0, state=None):
    """y: soma at 500 Hz; X: the observer's standardized synaptic filter bank (R1's matrix-neuron
    features). The observer knows its synaptic inputs and its probes, so it removes what they
    predict on average before estimating how the probe kernels change with the state; that removes
    variance, not signal, because the probe signs are independent of everything else."""
    from .observer import MatrixNeuron
    state = STATE if state is None else state
    n = len(y)
    a, b = int(round(500.0 / K.DT_OBS)), n                       # all of it after 0.5 s
    yc = y - y[a:b].mean()
    lin = MatrixNeuron().fit(X[a:b], yc[a:b])
    r0 = yc - lin.predict(X)                                     # background synaptic drive removed
    t, Y0 = _design(r0, a, b)
    xi_t = xi[t]
    Kp = (xi_t.T @ Y0) / (len(t) * eps)                         # pooled kernels
    r = r0 - _probe_prediction(xi, Kp, eps, n)                   # pooled probe responses removed
    _, Y = _design(r, a, b)
    Y = np.ascontiguousarray(Y, dtype=np.float32)               # the permutations dominate the cost
    # state labels: 'soma' = the observed soma voltage; 'branch' = each branch's linear synaptic
    # drive (the matrix neuron's branch component), a pair taking the branch of its first site
    if state == 'soma':
        label_sets = {0: bin_labels(y, t)}
        key = lambda i, j: 0
    else:
        label_sets = {}
        for br in sorted(set(int(v) for v in site_branch)):
            cols = np.nonzero(np.repeat(np.asarray(site_branch) == br, X.shape[1] // len(site_branch)))[0]
            label_sets[br] = bin_labels(X[:, cols] @ lin.w[cols], t)
        key = lambda i, j: int(site_branch[i])

    def kernels(perm=None):
        out = {}
        for k_, lab in label_sets.items():
            dev_, _ = binned_kernels(Y, xi_t, lab if perm is None else lab[perm], eps)
            out[k_] = Kp[None] + dev_
        return out

    Kb = kernels()
    pairs, wins = pair_windows(Kp)
    same = lambda i, j: site_branch[i] == site_branch[j]
    cross = lambda i, j: site_branch[i] != site_branch[j]
    sels = (('all', None), ('same', same), ('cross', cross))
    obs = {k: statistic(Kb, Kp, pairs, wins, sel, key) for k, sel in sels}
    rng = np.random.default_rng(seed)
    null = {k: [] for k in obs}
    for _ in range(n_perm):
        Kb_p = kernels(rng.permutation(len(t)))
        for k, sel in sels:
            null[k].append(statistic(Kb_p, Kp, pairs, wins, sel, key))
    res = {}
    for k in obs:
        v = np.array(null[k])
        mu, sd = float(v.mean()), float(v.std())
        res[k] = dict(S=obs[k], mu=mu, sd=sd, bar=mu + 3 * sd, z=(obs[k] - mu) / sd if sd > 0 else float('nan'),
                      rejected=bool(obs[k] >= mu + 3 * sd))
    out = dict(stat=res, window_lags=[int(w.sum()) for w in wins], bg_r2=float(
                   1 - np.var(r0[a:b]) / np.var(yc[a:b])),
               decay_ms_per_bin=[float(np.mean([decay_time(Kb[key(s, s)][bb, s]) for s in range(Kp.shape[0])]))
                                 for bb in range(N_BINS)],
               kernel_peak=[float(Kp[s].max()) for s in range(Kp.shape[0])])
    if xi_ref is not None:
        noise = (xi_ref[t].T @ Y.astype(float)) / (len(t) * eps)  # a kernel from signs never injected
        out['snr'] = [float(Kp[s].max() / noise[s].std()) for s in range(Kp.shape[0])]
    return out
