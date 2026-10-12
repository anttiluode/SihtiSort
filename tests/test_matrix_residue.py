import numpy as np

from sihtisort import cable as K
from sihtisort import observer as O


def _small_cell(seed=3):
    rng = np.random.default_rng(seed)
    morph = K.morphology(rng)
    n_sites = len(morph['sites'])
    return morph, rng, np.full(n_sites, 8.0), np.full(n_sites, 0.3)


def test_passive_cable_superposes_and_nmda_does_not():
    morph, rng, wa, wn = _small_cell()
    steps = 4000
    u1 = (rng.random((steps, len(wa))) < 0.004).astype(float)
    u2 = (rng.random((steps, len(wa))) < 0.004).astype(float)
    zero = np.zeros((steps, morph['n']))
    sim = lambda u, nmda: K.simulate(morph, u, wa, wn, zero, nmda)[0] - K.E_L
    lin_err = np.abs(sim(u1 + u2, False) - sim(u1, False) - sim(u2, False)).max()
    nmda_err = np.abs(sim(u1 + u2, True) - sim(u1, True) - sim(u2, True)).max()
    assert lin_err < 1e-9
    assert nmda_err > 1e-3


def test_ridge_cache_matches_direct_solve():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((3000, 40))
    y = Z[:, 0] - 0.5 * Z[:, 3] + rng.standard_normal(3000)
    w, b, lam = O.RidgeCV(Z).solve(y)
    Zc = Z - Z.mean(0)
    w2 = np.linalg.solve(Zc.T @ Zc + lam * np.eye(40), Zc.T @ (y - y.mean()))
    assert np.allclose(w, w2)
    assert np.isclose(b, y.mean() - Z.mean(0) @ w2)


def test_cascade_gradient():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6))
    y = np.tanh(2 * X[:, 0]) + 0.1 * rng.standard_normal(200)
    m = O.Cascade(n_hidden=3)
    m.nf = 6
    th = rng.standard_normal(6 * 3 + 3 + 3 * len(O.POST_TAUS) + 6 + 3) * 0.3
    _, g = m._loss(th, X, y)
    eps = 1e-6
    num = np.array([(m._loss(th + eps * e, X, y)[0] - m._loss(th - eps * e, X, y)[0]) / (2 * eps)
                    for e in np.eye(len(th))])
    assert np.abs(g - num).max() < 1e-6 * np.abs(num).max()


def test_phase_randomize_keeps_spectrum():
    rng = np.random.default_rng(2)
    e = np.cumsum(rng.standard_normal(1000)) * 0.1
    s = O.phase_randomize(e, rng)
    assert np.allclose(np.abs(np.fft.rfft(e)), np.abs(np.fft.rfft(s)))
    assert np.corrcoef(e, s)[0, 1] < 0.9
