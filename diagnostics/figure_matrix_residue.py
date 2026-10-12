"""Figure for Protocol R1: the same branch bursts seen by the tanh neuron (T1) and the NMDA
neuron (T2), with the fixed-filter cascade's prediction (fitted on regime A only).

    python diagnostics/figure_matrix_residue.py [seed]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sihtisort import cable as K
from sihtisort import observer as O


def main(seed=1000, t0=54.0, span=2.5):
    c = K.make_cell(seed)
    Xr = O.bank(c['counts'])
    W = K.windows()
    a, b = W['train']
    X = O.Standardizer(Xr[a:b])(Xr)
    t = np.arange(len(X)) * K.DT_OBS / 1000.0
    sel = (t >= t0) & (t < t0 + span)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 3, 3]))
    for tb, br in c['bursts']:
        if t0 <= tb / 1000 < t0 + span:
            axes[0].axvline(tb / 1000, color=f'C{br}', lw=2)
            axes[0].text(tb / 1000, 0.5, f' b{br}', color=f'C{br}', va='center')
    axes[0].set_yticks([])
    axes[0].set_title(f'Regime B, seed {seed}: branch bursts (never trained on)', loc='left', fontsize=10)
    for ax, truth, label in ((axes[1], 'T1', 'T1: passive cable read through tanh'),
                             (axes[2], 'T2', 'T2: NMDA dendrite')):
        y = c['y'][truth]
        lin = O.MatrixNeuron().fit(X[a:b], y[a:b])
        casc = O.Cascade().fit(X[a:b], y[a:b], lin=lin)
        P, L = casc.predict(X), lin.predict(X)
        e, f = W['B']
        ax.plot(t[sel], c['y_clean'][truth][sel] + 70, color='0.75', lw=4, label='noise-free truth')
        ax.plot(t[sel], y[sel] + 70, color='k', lw=0.6, label='recorded soma')
        ax.plot(t[sel], L[sel] + 70, color='C0', lw=1.2, label=f'matrix neuron (R² on B {O.r2(y[e:f], L[e:f]):.2f})')
        ax.plot(t[sel], P[sel] + 70, color='C3', lw=1.2, label=f'fixed-filter cascade (R² on B {O.r2(y[e:f], P[e:f]):.2f})')
        ax.set_ylabel('mV above rest')
        ax.set_title(label, loc='left', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
    axes[2].set_xlabel('time (s)')
    fig.tight_layout()
    out = os.path.join('results', 'matrix_residue_bursts.png')
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
