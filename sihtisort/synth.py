"""Synthetic multichannel recording with ground truth, following PROTOCOL.md.

Neurons have a spatial footprint on a linear probe and a biphasic waveform.
Mood neurons switch abruptly between two waveform states; the turn-taking
pair are two distinct neurons that fire in alternating blocks.
"""
import numpy as np

FS = 20000
N_CH = 16
DURATION = 60.0
DEAD = 0.002
WIN_PRE, WIN_POST = 16, 32          # snippet window in samples (-0.8 .. +1.6 ms)
T_WAVE = np.arange(-WIN_PRE, WIN_POST) / FS * 1000.0   # ms


def waveform(A, w, r):
    """Biphasic spike: trough of depth A and width w (ms), repolarization peak A*r."""
    t = T_WAVE
    trough = -A * np.exp(-t ** 2 / (2 * w ** 2))
    peak = A * r * np.exp(-(t - 3.2 * w) ** 2 / (2 * (2.5 * w) ** 2))
    return trough + peak


def footprint(pos, sigma):
    c = np.arange(N_CH)
    return np.exp(-(c - pos) ** 2 / (2 * sigma ** 2))


def template(A, w, r, pos, sigma):
    """(samples, channels) template; A is the best-channel trough depth."""
    return np.outer(waveform(A, w, r), footprint(pos, sigma))


def block_process(rng, duration, mean_dwell=8.0, min_dwell=1.0):
    """Alternating two-state blocks: list of (start, end, state)."""
    t, s, out = 0.0, int(rng.integers(2)), []
    while t < duration:
        d = max(min_dwell, rng.exponential(mean_dwell))
        out.append((t, min(duration, t + d), s))
        t, s = t + d, 1 - s
    return out


def poisson_train(rng, rate, duration, windows=None):
    """Poisson spikes with dead time; if windows given, keep only spikes inside them."""
    t, out = 0.0, []
    while True:
        t += DEAD + rng.exponential(1.0 / rate)
        if t >= duration:
            break
        out.append(t)
    out = np.array(out)
    if windows is not None:
        keep = np.zeros(len(out), bool)
        for a, b in windows:
            keep |= (out >= a) & (out < b)
        out = out[keep]
    return out


def state_at(times, blocks):
    st = np.zeros(len(times), int)
    for a, b, s in blocks:
        st[(times >= a) & (times < b)] = s
    return st


def make_recording(seed, duration=DURATION, min_sep=None, turn_range=None):
    """min_sep (channels): if given, POST HOC sparse setting -- every neuron outside the
    turn-taking pair sits at least min_sep from every other neuron. Default None is the
    protocol's setting (positions uniformly random)."""
    rng = np.random.default_rng(seed)
    n = int(duration * FS)
    neurons = []
    kinds = ['mood_strong'] * 2 + ['mood_mild'] * 2 + ['turn_a', 'turn_b'] + ['stable'] * 4
    turn_pos = rng.uniform(2.0, N_CH - 3.0)
    turn_offset = rng.uniform(0.5, 2.0)
    if turn_range is not None:     # PROTOCOL_2 identifiability set: same draw, mapped onto a closer range
        lo, hi = turn_range
        turn_offset = lo + (turn_offset - 0.5) / 1.5 * (hi - lo)
    turn_blocks = block_process(rng, duration)
    for k, kind in enumerate(kinds):
        sigma = rng.uniform(1.2, 1.8)
        A = rng.uniform(9, 15) if kind.startswith('mood') else rng.uniform(7, 14)
        w = rng.uniform(0.10, 0.18)
        r = rng.uniform(0.2, 0.45)
        rate = rng.uniform(3, 12)
        if kind == 'turn_a':
            pos = turn_pos
        elif kind == 'turn_b':
            pos = turn_pos + turn_offset
        else:
            pos = rng.uniform(0.5, N_CH - 1.5)
            if min_sep is not None:
                taken = [turn_pos, turn_pos + turn_offset] + [nu['pos'] for nu in neurons]
                for _ in range(10000):
                    if min(abs(pos - q) for q in taken) >= min_sep:
                        break
                    pos = rng.uniform(0.5, N_CH - 1.5)
                else:
                    raise RuntimeError('could not place neuron with min_sep')
        states = [template(A, w, r, pos, sigma)]
        if kind == 'mood_strong':
            states.append(template(0.6 * A, 1.6 * w, 1.3 * r, pos, sigma * (1 + rng.uniform(-0.15, 0.15))))
        elif kind == 'mood_mild':
            states.append(template(0.8 * A, 1.3 * w, 1.1 * r, pos, sigma * (1 + rng.uniform(-0.15, 0.15))))
        if kind.startswith('mood'):
            blocks = block_process(rng, duration)
            times = poisson_train(rng, rate, duration)
            st = state_at(times, blocks)
        elif kind in ('turn_a', 'turn_b'):
            want = 0 if kind == 'turn_a' else 1
            times = poisson_train(rng, rate, duration,
                                  [(a, b) for a, b, s in turn_blocks if s == want])
            st = np.zeros(len(times), int)
        else:
            times = poisson_train(rng, rate, duration)
            st = np.zeros(len(times), int)
        neurons.append(dict(id=k, kind=kind, pos=pos, sigma=sigma, rate=rate,
                            templates=states, times=times, states=st))

    # noise: white, smoothed across channels and lightly in time, unit std
    white = rng.standard_normal((n + 1, N_CH + 2)).astype(np.float32)
    noise = 0.25 * white[:, :-2] + 0.5 * white[:, 1:-1] + 0.25 * white[:, 2:]   # across channels
    noise = 0.6 * noise[1:] + 0.4 * noise[:-1]                                  # lightly in time
    noise /= noise.std(0, keepdims=True)
    data = noise
    gt_t, gt_u, gt_s = [], [], []
    for nu in neurons:
        samples = np.round(nu['times'] * FS).astype(int)
        jit = 1 + 0.07 * rng.standard_normal(len(samples))
        for smp, s, j in zip(samples, nu['states'], jit):
            a, b = smp - WIN_PRE, smp + WIN_POST
            if a < 0 or b > n:
                continue
            data[a:b] += (j * nu['templates'][s]).astype(np.float32)
            gt_t.append(smp); gt_u.append(nu['id']); gt_s.append(s)
    order = np.argsort(gt_t)
    gt = dict(t=np.array(gt_t)[order], unit=np.array(gt_u)[order], state=np.array(gt_s)[order])
    meta = dict(seed=seed, turn_offset=float(turn_offset),
                neurons=[dict(id=nu['id'], kind=nu['kind'], pos=float(nu['pos']),
                              rate=float(nu['rate']), n=int(len(nu['times'])))
                         for nu in neurons])
    return data, gt, meta
