import numpy as np
from scipy.signal import lfilter

from sihtisort import probe as PR


def _world(premerge, T=120_000, eps=0.3, seed=0):
    """Two probed sites and a slow background that moves the operating point of a tanh."""
    rng = np.random.default_rng(seed)
    xi = rng.choice([-1.0, 1.0], size=(T, 2))
    bg = lfilter([0.02], [1, -0.98], rng.standard_normal(T)) * 6.0
    h1 = lfilter([0.3], [1, -0.7], eps * xi[:, 0])
    h2 = lfilter([0.15], [1, -0.85], eps * xi[:, 1])
    h1 = np.concatenate([[0.0], h1[:-1]])                     # the probe acts one step late
    h2 = np.concatenate([[0.0], h2[:-1]])
    if premerge:
        y = np.tanh(h1 + bg) + h2                             # site 2 bypasses the gain
    else:
        y = np.tanh(h1 + h2 + bg)                             # one pointwise gain after merging
    return y + 0.01 * rng.standard_normal(T), xi, bg


def _z(premerge):
    y, xi, bg = _world(premerge)
    t, Y = PR._design(y - y.mean(), 0, len(y))
    labels = PR.bin_labels(bg, t)
    Kb, Kp = PR.binned_kernels(Y, xi[t], labels, 0.3)
    pairs, wins = PR.pair_windows(Kp)
    s_obs = PR.statistic(Kb, Kp, pairs, wins)
    rng = np.random.default_rng(1)
    null = [PR.statistic(PR.binned_kernels(Y, xi[t], rng.permutation(labels), 0.3)[0], Kp, pairs, wins)
            for _ in range(60)]
    return (s_obs - np.mean(null)) / np.std(null)


def test_common_pointwise_gain_keeps_the_invariant():
    assert _z(premerge=False) < 3


def test_gain_on_one_path_breaks_it():
    assert _z(premerge=True) > 3
