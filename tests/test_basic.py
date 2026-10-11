import numpy as np
from sihtisort.synth import make_recording
from sihtisort.sort import switch_ratio, viterbi_states, viol
from sihtisort.evaluate import match_idx


def test_recording_is_deterministic():
    a = make_recording(7, duration=2.0)
    b = make_recording(7, duration=2.0)
    assert np.array_equal(a[1]['t'], b[1]['t']) and np.allclose(a[0], b[0])


def test_switch_ratio_blocks_vs_interleaved():
    rng = np.random.default_rng(0)
    blocks_a = np.concatenate([np.arange(0, 100), np.arange(200, 300)]) * 1000
    blocks_b = np.concatenate([np.arange(100, 200), np.arange(300, 400)]) * 1000
    assert switch_ratio(blocks_a, blocks_b) < 0.05
    t = np.sort(rng.choice(10 ** 6, 400, replace=False))
    lab = rng.integers(0, 2, 400)
    assert 0.8 < switch_ratio(t[lab == 0], t[lab == 1]) < 1.2


def test_viol_counts_short_isis():
    assert viol(np.array([0, 10, 1000, 2000])) == 1 / 3


def test_viterbi_smooths_a_flip():
    t = np.arange(10) * 100
    err = [[0.0, 4.0]] * 10
    err[5] = [1.0, 0.0]            # weak evidence for state 1 in the middle
    assert list(viterbi_states(t, err, 2)) == [0] * 10


def test_match_idx_tolerance():
    assert list(match_idx(np.array([100, 200]), np.array([105, 300]))) == [0, -1]
