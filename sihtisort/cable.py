"""A small branched neuron observed at the soma, following PROTOCOL_R1.md.

Three truths share one seed (morphology, synapses, input trains, noise):
  T0  passive cable                          -- exactly a matrix neuron
  T1  passive cable read through tanh         -- fixed filter -> static nonlinearity
  T2  cable with NMDA conductances (Mg block) -- nonlinearity inside the operator

The cable is integrated with implicit Euler: (C/dt + G) v[t+1] = C/dt v[t] + I[t].
For T0 this is literally v[t+1] = P v[t] + X I[t] with P = (C/dt + G)^-1 C/dt.
"""
import numpy as np

DT = 0.25            # ms, integration step
OBS = 8              # observe every 8th step -> 500 Hz (2 ms)
DT_OBS = DT * OBS
DURATION = 60_000.0  # ms
A_END, HOLD_END = 40_000.0, 50_000.0   # training 0.5-40 s, held-out 40.5-50 s, regime B 50.5-60 s
E_L, E_NMDA = -70.0, 0.0
TAU_M = 20.0
TAU_AMPA = 2.0
TAU_NR, TAU_ND = 2.0, 60.0
N_BRANCH, SITES_PER_BRANCH = 4, 2

# Tuned on calibration seeds 900-902 only (see PROTOCOL_R1.md, deviations).
PARAMS = dict(
    g_ax=2.0,          # axial conductance between neighbours (per ms, C=1 units), times U(0.5, 2) per branch
    c_soma=5.0,       # soma capacitance relative to a dendritic compartment
    w_ampa=8.0,       # AMPA current weight (mV/ms per spike at C=1), times U(0.6, 1.4) per site
    w_nmda=0.30,      # NMDA peak conductance per spike (per ms), times U(0.6, 1.4) per site
    rate_a=6.0,        # Hz per site in regime A
    burst_every=300.0,  # ms, mean interval between branch bursts in regime B
    burst_spikes=4,    # spikes per site per burst
    burst_window=15.0,  # ms
    ou_sd=0.02,        # OU current noise sd per compartment (mV/ms)
    ou_tau=5.0,        # ms
    meas_sd=0.05,      # mV, soma measurement noise
    tanh_rel=2.0,      # T1 saturation scale, in units of the cell's noise-free passive soma sd
)


def morphology(rng, p=PARAMS):
    """Soma (index 0) plus N_BRANCH unbranched dendrites attached to it."""
    lengths = rng.integers(8, 17, N_BRANCH)
    gax = p['g_ax'] * rng.uniform(0.5, 2.0, N_BRANCH)
    n = 1 + int(lengths.sum())
    C = np.ones(n)
    C[0] = p['c_soma']
    G = np.diag(C / TAU_M)                       # leak, scaled by capacitance
    branch_of = np.full(n, -1)
    sites, site_branch = [], []
    idx = 1
    for b in range(N_BRANCH):
        comps = list(range(idx, idx + lengths[b]))
        branch_of[comps] = b
        prev = 0
        for c in comps:
            G[prev, prev] += gax[b]; G[c, c] += gax[b]
            G[prev, c] -= gax[b]; G[c, prev] -= gax[b]
            prev = c
        distal = comps[len(comps) // 2:]
        for c in rng.choice(distal, SITES_PER_BRANCH, replace=False):
            sites.append(int(c)); site_branch.append(b)
        idx += lengths[b]
    return dict(C=C, G=G, n=n, sites=np.array(sites), site_branch=np.array(site_branch),
                lengths=lengths, gax=gax, branch_of=branch_of)


def input_trains(rng, p=PARAMS):
    """Spike counts per integration step and site (steps, sites)."""
    steps = int(DURATION / DT)
    n_sites = N_BRANCH * SITES_PER_BRANCH
    t = np.arange(steps) * DT
    rate = np.where(t < HOLD_END, p['rate_a'], p['rate_a'] / 2.0)       # Hz
    counts = rng.poisson((rate * DT / 1000.0)[:, None], (steps, n_sites)).astype(float)
    # regime B: branch bursts
    tb = HOLD_END + rng.exponential(p['burst_every'])
    bursts = []
    while tb < DURATION - p['burst_window']:
        b = int(rng.integers(N_BRANCH))
        for s in range(b * SITES_PER_BRANCH, (b + 1) * SITES_PER_BRANCH):
            ts = tb + rng.uniform(0, p['burst_window'], p['burst_spikes'])
            np.add.at(counts[:, s], (ts / DT).astype(int), 1.0)
        bursts.append((tb, b))
        tb += p['burst_window'] + rng.exponential(p['burst_every'])
    return counts, bursts


def simulate(morph, counts, w_ampa, w_nmda, ou, nmda):
    """Integrate the cable. Returns soma voltage at every step and site voltages at the observation rate."""
    C, G, n, sites = morph['C'], morph['G'], morph['n'], morph['sites']
    steps = counts.shape[0]
    Cdt = C / DT
    M = np.linalg.inv(np.diag(Cdt) + G)
    # AMPA current at each site: exponential kernel driven by spike counts
    decay = np.exp(-DT / TAU_AMPA)
    from scipy.signal import lfilter
    i_ampa = lfilter([1.0], [1.0, -decay], counts, axis=0) * w_ampa[None, :]
    v = np.full(n, E_L)
    soma = np.empty(steps)
    site_v = np.empty((steps // OBS, len(sites)))
    dr, dd = np.exp(-DT / TAU_NR), np.exp(-DT / TAU_ND)
    gr = np.zeros(len(sites)); gd = np.zeros(len(sites))
    norm = 1.0 / (np.exp(-np.log(TAU_ND / TAU_NR) * TAU_NR / (TAU_ND - TAU_NR))
                  - np.exp(-np.log(TAU_ND / TAU_NR) * TAU_ND / (TAU_ND - TAU_NR)))   # peak of g = 1 per spike
    I = np.empty(n)
    for t in range(steps):
        soma[t] = v[0]
        if t % OBS == 0:
            site_v[t // OBS] = v[sites]
        I[:] = ou[t]
        I[sites] += i_ampa[t]
        if nmda:
            gr = gr * dr + counts[t] * w_nmda * norm
            gd = gd * dd + counts[t] * w_nmda * norm
            vs = v[sites]
            B = 1.0 / (1.0 + np.exp(-0.062 * vs) / 3.57)       # Mg2+ = 1 mM (Jahr & Stevens)
            I[sites] += (gd - gr) * B * (E_NMDA - vs)
        v = M @ (Cdt * v + I + C * E_L / TAU_M)          # leak toward E_L
    return soma, site_v


def make_cell(seed, p=PARAMS):
    """Simulate one seed under all three truths. Observations at 500 Hz."""
    rng = np.random.default_rng(seed)
    morph = morphology(rng, p)
    n_sites = len(morph['sites'])
    w_ampa = p['w_ampa'] * rng.uniform(0.6, 1.4, n_sites)
    w_nmda = p['w_nmda'] * rng.uniform(0.6, 1.4, n_sites)
    counts, bursts = input_trains(rng, p)
    steps = counts.shape[0]
    # OU current noise in every compartment (same realisation for every truth)
    a = np.exp(-DT / p['ou_tau'])
    from scipy.signal import lfilter
    white = rng.standard_normal((steps, morph['n'])) * p['ou_sd'] * np.sqrt(1 - a * a)
    ou = lfilter([1.0], [1.0, -a], white, axis=0)
    meas = rng.standard_normal(steps // OBS) * p['meas_sd']
    soma0, sv0 = simulate(morph, counts, w_ampa, w_nmda, ou, nmda=False)
    soma2, sv2 = simulate(morph, counts, w_ampa, w_nmda, ou, nmda=True)
    v0 = soma0[::OBS]
    v2 = soma2[::OBS]
    # noise-free versions (same inputs, no OU current, no measurement noise): a reference for
    # how much of each window is predictable from the inputs at all
    zero = np.zeros_like(ou)
    c0 = simulate(morph, counts, w_ampa, w_nmda, zero, nmda=False)[0][::OBS]
    c2 = simulate(morph, counts, w_ampa, w_nmda, zero, nmda=True)[0][::OBS]
    s = p['tanh_rel'] * float(np.std(c0))      # every cell gets a comparable saturation
    y = {'T0': v0 + meas,
         'T1': E_L + s * np.tanh((v0 - E_L) / s) + meas,
         'T2': v2 + meas}
    y_clean = {'T0': c0, 'T1': E_L + s * np.tanh((c0 - E_L) / s), 'T2': c2}
    return dict(y=y, y_clean=y_clean, counts=counts, site_branch=morph['site_branch'], bursts=bursts,
                site_v={'T0': sv0, 'T2': sv2}, morph=dict(lengths=morph['lengths'].tolist(),
                gax=morph['gax'].tolist(), sites=morph['sites'].tolist()),
                w_ampa=w_ampa, w_nmda=w_nmda)


def windows():
    """Observation-sample index ranges for training, held-out and regime B."""
    ms = lambda t: int(round(t / DT_OBS))
    return dict(train=(ms(500), ms(A_END)), hold=(ms(A_END + 500), ms(HOLD_END)),
                B=(ms(HOLD_END + 500), ms(DURATION)))
