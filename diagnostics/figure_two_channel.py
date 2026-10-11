"""Figure: the turn-taking pair, merged or apart, at each moment of the online run (seeds 50-61)."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import ListedColormap        # noqa: E402
import numpy as np                                  # noqa: E402

rows = json.load(open('results/two_channel.json'))['identifiability']['rows']
rows = sorted(rows, key=lambda r: r['turn_offset'])
T = [p['T'] for p in rows[0]['trace']]
grids = {k: np.array([[p[k]['turn_merged'] for p in r['trace']] for r in rows], float)
         for k in ('committed', 'two_channel')}
cmap = ListedColormap(['#4f7fbf', '#e8a33d'])     # apart, merged
fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2), sharey=True)
titles = {'committed': 'Committed sorter only', 'two_channel': 'With the remainder (two channels)'}
for ax, k in zip(axes, grids):
    ax.imshow(grids[k], cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(T))); ax.set_xticklabels([f'{t}s' for t in T])
    ax.set_title(titles[k], loc='left', fontsize=10)
    ax.set_xlabel('recording seen so far')
    for i in range(len(rows)):
        for j in range(len(T)):
            ax.text(j, i, 'merged' if grids[k][i, j] else 'apart', ha='center', va='center', fontsize=7,
                    color='black' if grids[k][i, j] else 'white')
axes[0].set_yticks(range(len(rows)))
axes[0].set_yticklabels([f"{r['turn_offset']:.2f} ch, rates {r['turn_rate_ratio']:.2f}x" for r in rows], fontsize=8)
axes[0].set_ylabel('turn-taking pair: true distance apart, rate ratio')
fig.suptitle('Two neurons taking turns: does the sorter keep them apart as the recording grows?', fontsize=11, x=0.02, ha='left')
fig.tight_layout()
fig.savefig('results/two_channel_online.png', dpi=130)
