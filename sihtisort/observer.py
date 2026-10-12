"""Observers of a soma-recorded neuron, their residues, and the calibrated null (PROTOCOL_R1.md).

M_lin  the matrix neuron: causal exponential filter bank on each site's spikes, ridge to soma voltage.
M_LN   the best fixed-filter cascade: the same bank -> tanh hidden units -> fixed exponential
       post-filters -> readout, plus the linear passthrough and a cubic output polynomial.
P_gen  generic residue probe: fixed random tanh features of the bank, ridge.
P_state  the bet's own form: each branch's features times that branch's linear drive, ridge.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from .cable import DT, DT_OBS, OBS

TAUS = np.geomspace(0.5, 300.0, 16)        # ms, input bank
POST_TAUS = [0.0, 2.0, 8.0, 30.0, 100.0]   # ms; 0 = instantaneous
N_HIDDEN = 8
N_GEN = 1000
GEN_GAIN = 0.5
LBFGS_ITERS = 1000
L2 = 1e-3
LAMBDA_GRID = np.geomspace(1e-6, 10.0, 15)  # times the number of samples
VAL_FRAC = 0.25                             # cascade early-stopping split (end of training)
PATIENCE = 200                              # stop after this many iterations without improvement


# ---------------------------------------------------------------- features
def bank(counts):
    """(T_obs, sites * len(TAUS)) causal exponential features; spikes enter one step late."""
    c = np.vstack([np.zeros((1, counts.shape[1])), counts[:-1]])
    cols = []
    for s in range(counts.shape[1]):
        for tau in TAUS:
            a = np.exp(-DT / tau)
            cols.append(lfilter([1.0 - a], [1.0, -a], c[:, s])[::OBS])
    return np.stack(cols, 1)


def branch_cols(site_branch):
    nt = len(TAUS)
    out = {}
    for s, b in enumerate(site_branch):
        out.setdefault(int(b), []).extend(range(s * nt, (s + 1) * nt))
    return {b: np.array(v) for b, v in out.items()}


class Standardizer:
    def __init__(self, X):
        self.m = X.mean(0)
        self.s = X.std(0) + 1e-12

    def __call__(self, X):
        return (X - self.m) / self.s


# ---------------------------------------------------------------- ridge with blocked CV
class RidgeCV:
    """Ridge with intercept; lambda chosen by contiguous-block CV. The decompositions depend only
    on the features, so one instance serves many targets (the null draws reuse it)."""

    def __init__(self, Z, folds=5):
        T, F = Z.shape
        self.T, self.F = T, F
        self.Zm = Z.mean(0)
        self.Zc = Z - self.Zm
        self.Gt = self.Zc.T @ self.Zc
        self.evT, self.VT = np.linalg.eigh(self.Gt)
        edges = np.linspace(0, T, folds + 1).astype(int)
        self.folds = []
        for k in range(folds):
            a, b = edges[k], edges[k + 1]
            Zf = self.Zc[a:b]
            ev, V = np.linalg.eigh(self.Gt - Zf.T @ Zf)
            self.folds.append((a, b, ev, V, Zf @ V))

    def solve(self, y):
        yc = y - y.mean()
        rt = self.Zc.T @ yc
        sse = np.zeros(len(LAMBDA_GRID))
        for a, b, ev, V, ZV in self.folds:
            yf = yc[a:b]
            Vr = V.T @ (rt - self.Zc[a:b].T @ yf)
            for i, lam in enumerate(LAMBDA_GRID * (self.T - (b - a))):
                sse[i] += np.sum((yf - ZV @ (Vr / (ev + lam))) ** 2)
        lam = LAMBDA_GRID[int(np.argmin(sse))] * self.T
        w = self.VT @ ((self.VT.T @ rt) / (self.evT + lam))
        return w, y.mean() - self.Zm @ w, lam


def ridge_cv(Z, y, folds=5):
    return RidgeCV(Z, folds).solve(y)


def r2(y, yhat):
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


# ---------------------------------------------------------------- M_lin
class MatrixNeuron:
    def fit(self, X, y):
        self.w, self.b, self.lam = ridge_cv(X, y)
        return self

    def predict(self, X):
        return X @ self.w + self.b


# ---------------------------------------------------------------- M_LN
def _filters():
    out = []
    for tau in POST_TAUS:
        if tau == 0:
            out.append(None)
        else:
            a = np.exp(-DT_OBS / tau)
            out.append(([1.0 - a], [1.0, -a]))
    return out


def _filt(f, x):
    return x if f is None else lfilter(f[0], f[1], x, axis=0)


def _filt_adj(f, g):
    return g if f is None else lfilter(f[0], f[1], g[::-1], axis=0)[::-1]


class Cascade:
    """Fixed filters, nonlinearity only between them. Fitted by L-BFGS from a fixed start."""

    def __init__(self, n_hidden=N_HIDDEN, iters=LBFGS_ITERS, l2=L2, seed=12345):
        self.H, self.iters, self.l2, self.seed = n_hidden, iters, l2, seed
        self.F = _filters()

    def _unpack(self, th):
        F, H, K = self.nf, self.H, len(self.F)
        i = 0
        W = th[i:i + F * H].reshape(F, H); i += F * H
        b = th[i:i + H]; i += H
        R = th[i:i + H * K].reshape(H, K); i += H * K
        wl = th[i:i + F]; i += F
        c, q2, q3 = th[i], th[i + 1], th[i + 2]
        return W, b, R, wl, c, q2, q3

    def _forward(self, th, X):
        W, b, R, wl, c, q2, q3 = self._unpack(th)
        a = np.tanh(X @ W + b)
        # filtering is linear, so filter each readout channel instead of every hidden unit
        u = c + X @ wl + sum(_filt(f, a @ R[:, k]) for k, f in enumerate(self.F))
        yh = u + q2 * u ** 2 + q3 * u ** 3
        return yh, (a, u)

    def _loss(self, th, X, y):
        W, b, R, wl, c, q2, q3 = self._unpack(th)
        T = len(y)
        yh, (a, u) = self._forward(th, X)
        r = yh - y
        loss = 0.5 * np.mean(r ** 2) + 0.5 * self.l2 * (np.sum(W ** 2) + np.sum(R ** 2))
        gy = r / T
        gu = gy * (1 + 2 * q2 * u + 3 * q3 * u ** 2)
        gq2, gq3 = gy @ (u ** 2), gy @ (u ** 3)
        gc = gu.sum()
        gwl = X.T @ gu
        Hk = np.stack([_filt_adj(f, gu) for f in self.F], 1)      # (T, K) adjoint-filtered gradient
        gR = a.T @ Hk + self.l2 * R
        ga = Hk @ R.T
        gz = ga * (1 - a ** 2)
        gW = X.T @ gz + self.l2 * W
        gb = gz.sum(0)
        g = np.concatenate([gW.ravel(), gb, gR.ravel(), gwl, [gc, gq2, gq3]])
        return loss, g

    def fit(self, X, y, lin=None):
        self.nf = X.shape[1]
        self.ym, self.ys = y.mean(), y.std()
        ys = (y - self.ym) / self.ys
        rng = np.random.default_rng(self.seed)
        W0 = rng.standard_normal((self.nf, self.H)) * (0.5 / np.sqrt(self.nf))
        b0 = np.zeros(self.H)
        R0 = np.zeros((self.H, len(self.F)))
        if lin is None:
            lin = MatrixNeuron().fit(X, y)
        wl0 = lin.w / self.ys
        c0 = (lin.b - self.ym) / self.ys
        th0 = np.concatenate([W0.ravel(), b0, R0.ravel(), wl0, [c0, 0.0, 0.0]])
        # early stopping: fit on the first (1 - VAL_FRAC) of training, keep the iterate that
        # predicts the last VAL_FRAC best (the forward pass runs over all of training, so the
        # filters are warm at the split)
        n_fit = int(len(ys) * (1 - VAL_FRAC))
        best = [np.inf, th0, 0]
        it = [0]

        def watch(xk):
            it[0] += 1
            v = np.mean((self._forward(xk, X)[0][n_fit:] - ys[n_fit:]) ** 2)
            if v < best[0]:
                best[:] = [v, xk.copy(), it[0]]
            elif it[0] - best[2] >= PATIENCE:
                raise StopIteration

        watch(th0)
        it[0] = 0
        res = minimize(self._loss, th0, args=(X[:n_fit], ys[:n_fit]), jac=True, method='L-BFGS-B',
                       callback=watch, options=dict(maxiter=self.iters, maxcor=20))
        self.th, self.nit, self.best_it = best[1], res.nit, best[2]
        self.val_mse = float(best[0])
        return self

    def predict(self, X):
        return self._forward(self.th, X)[0] * self.ys + self.ym


# ---------------------------------------------------------------- residue probes
class GenProbe:
    """Fixed random tanh features of the bank. The features do not depend on any fit, so they and
    their ridge decompositions are built once per recording and shared by every arm and null draw."""

    def __init__(self, X, win, seed=777):
        nf = X.shape[1]
        rng = np.random.default_rng(seed)
        Rg = rng.standard_normal((nf, N_GEN)) * (GEN_GAIN / np.sqrt(nf))
        bg = rng.uniform(-1, 1, N_GEN) * GEN_GAIN
        self.Z = {k: np.tanh(X[a:b] @ Rg + bg) for k, (a, b) in win.items()}
        self.cv = RidgeCV(self.Z['train'])

    def fit(self, e_train):
        return self.cv.solve(e_train)

    def predict(self, wb, window):
        w, b, _ = wb
        return self.Z[window] @ w + b


class StateProbe:
    """The bet's own form: each branch's features times that branch's linear drive (from the
    matrix neuron fitted to the same data), ridge to the residue."""

    def __init__(self, cols):
        self.cols = cols

    def feats(self, X, lin):
        out = []
        for b, cols in self.cols.items():
            z = X[:, cols] @ lin.w[cols]
            out.append(X[:, cols] * (z / self.zs[b])[:, None])
        return np.concatenate(out, 1)

    def fit(self, X, lin, e):
        self.zs = {b: (X[:, c] @ lin.w[c]).std() + 1e-12 for b, c in self.cols.items()}
        self.w, self.b, self.lam = ridge_cv(self.feats(X, lin), e)
        return self

    def predict(self, X, lin):
        return self.feats(X, lin) @ self.w + self.b


def phase_randomize(e, rng):
    """Same power spectrum, random phases."""
    Fe = np.fft.rfft(e)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Fe)))
    ph[0] = 1.0
    if len(e) % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(Fe * ph, n=len(e))


# ---------------------------------------------------------------- one arm
def fit_all(model_name, X, ytr, win, cols, gen):
    """Fit the model on training only; return the matrix neuron, the model's prediction over the
    whole recording (so the cascade's filters are warm in every window) and both probes."""
    a, b = win['train']
    Xtr = X[a:b]
    lin = MatrixNeuron().fit(Xtr, ytr)
    model = lin if model_name == 'lin' else Cascade().fit(Xtr, ytr, lin=lin)
    P = model.predict(X)
    etr = ytr - P[a:b]
    probes = dict(gen=gen.fit(etr), state=StateProbe(cols).fit(Xtr, lin, etr))
    return lin, model, P, probes


def probe_pred(probes, gen, lin, X, win, window):
    a, b = win[window]
    return dict(gen=gen.predict(probes['gen'], window), state=probes['state'].predict(X[a:b], lin))


def structure(lin, P, probes, gen, X, y_hold, win):
    c, d = win['hold']
    e = y_hold - P[c:d]
    return {k: r2(e, v) for k, v in probe_pred(probes, gen, lin, X, win, 'hold').items()}


def run_arm(model_name, X, y, win, cols, gen, n_null=20, seed=0, y_clean=None):
    """Fit on training, score structure on held-out, transfer to regime B, calibrated null."""
    (a, b), (c, d), (e, f) = win['train'], win['hold'], win['B']
    lin, model, P, probes = fit_all(model_name, X, y[a:b], win, cols, gen)
    S = structure(lin, P, probes, gen, X, y[c:d], win)
    yho, yB = y[c:d], y[e:f]
    ph = probe_pred(probes, gen, lin, X, win, 'hold')
    pB = probe_pred(probes, gen, lin, X, win, 'B')
    out = dict(S=S,
               r2_hold=r2(yho, P[c:d]), r2_B=r2(yB, P[e:f]),
               r2_hold_plus={k: r2(yho, P[c:d] + v) for k, v in ph.items()},
               r2_B_plus={k: r2(yB, P[e:f] + v) for k, v in pB.items()},
               lam=float(lin.lam))
    if y_clean is not None:
        out['ceiling_hold'] = r2(yho, y_clean[c:d])
        out['ceiling_B'] = r2(yB, y_clean[e:f])
    if model_name != 'lin':
        out['nit'], out['best_it'] = int(model.nit), int(model.best_it)
    etr, eho = y[a:b] - P[a:b], yho - P[c:d]
    rng = np.random.default_rng(seed)
    nulls, naive = [], []
    for _ in range(n_null):
        ntr, nho = phase_randomize(etr, rng), phase_randomize(eho, rng)
        # naive: the real fit, the probes asked to find input structure in spectrum-matched noise
        pn = dict(gen=gen.fit(ntr), state=StateProbe(cols).fit(X[a:b], lin, ntr))
        naive.append({k: r2(nho, v) for k, v in probe_pred(pn, gen, lin, X, win, 'hold').items()})
        # calibrated: the identical procedure, refitted on data the fitted model generates
        ys = P.copy()
        ys[a:b] += ntr
        ys[c:d] += nho
        l2, m2, P2, p2 = fit_all(model_name, X, ys[a:b], win, cols, gen)
        nulls.append(structure(l2, P2, p2, gen, X, ys[c:d], win))
    out['null'] = nulls
    out['naive'] = naive
    for k in ('gen', 'state'):
        for name, draws in (('cal', nulls), ('naive', naive)):
            v = np.array([dd[k] for dd in draws]) if draws else np.array([np.nan])
            bar = max(0.01, v.mean() + 3 * v.std())
            out[f'{name}_{k}'] = dict(mu=float(v.mean()), sd=float(v.std()), bar=float(bar),
                                      structured=bool(S[k] >= bar))
    return out
