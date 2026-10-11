"""POST HOC: split the test-set mood neurons by how close their nearest other neuron sits."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihtisort.synth import make_recording   # noqa: E402

r = json.load(open('results/test.json'))
rows = []
for seed, res in r['per_seed'].items():
    _, _, meta = make_recording(int(seed))
    pos = {n['id']: n['pos'] for n in meta['neurons']}
    for i, n in enumerate(res['S']['neurons']):
        if not n['kind'].startswith('mood'):
            continue
        d = min(abs(pos[n['id']] - p) for k, p in pos.items() if k != n['id'])
        rows.append(dict(seed=int(seed), id=n['id'], kind=n['kind'], nearest=round(d, 2),
                         **{v: res[v]['neurons'][i]['one_unit'] for v in ('B1', 'B2', 'S', 'S-block')}))
out = {}
for name, f in [('crowded (< 0.5 ch)', lambda x: x['nearest'] < 0.5), ('isolated (>= 0.5 ch)', lambda x: x['nearest'] >= 0.5)]:
    sel = [x for x in rows if f(x)]
    out[name] = dict(n=len(sel), **{v: sum(x[v] for x in sel) for v in ('B1', 'B2', 'S', 'S-block')})
print(json.dumps(out, indent=1))
json.dump(dict(note='POST HOC', summary=out, rows=rows), open('results/crowding_test.json', 'w'), indent=1)
