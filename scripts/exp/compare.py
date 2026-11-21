import sys, glob, re; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np
from tab import score
from offline import agg, decide
D = 'data/model/relbench_public/err_study/'
tags = sys.argv[1:]
for val in ('2026-10-25', '2026-11-01'):
    print('val', val)
    for tag in tags:
        res = {}
        for f in sorted(glob.glob(f'{D}{tag}_val{val}_s*.probs.npz')):
            z = np.load(f); yv, yh = z['val_y'].astype(int), z['test_y'].astype(int)
            for how, rule in (('max', 'valthr'), ('noisyor', 'valthr'), ('mean', 'quota1.0'), ('max', 'oracle_k')):
                res.setdefault(f'{how}/{rule}', []).append(score(yh, decide(rule, agg(z['val_daily'], how), agg(z['test_daily'], how), yv, yh))[0])
        print(f'  {tag:8s} n={len(next(iter(res.values()), []))}', '  '.join(f'{k}: {np.mean(v):.4f}' for k, v in res.items()))
