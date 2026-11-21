"""Offline decision-rule study on saved GNN probabilities (val + holdout), multi-seed."""
import sys, glob, re; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np
from tab import score, best_thr
D = 'data/model/relbench_public/err_study/'
def load():
    runs = {}
    for f in sorted(glob.glob(D + 'h3_val*_s*.probs.npz')):
        m = re.search(r'val(\S+)_s(\d+)\.probs', f); z = np.load(f)
        runs.setdefault(m.group(1), {})[int(m.group(2))] = {k: z[k] for k in z.files}
    return runs
def agg(daily, how):
    if how == 'max': return daily.max(1)
    if how == 'noisyor': return 1 - np.prod(1 - daily, axis=1)
    if how == 'mean': return daily.mean(1)
    if how == 'sum': return daily.sum(1)
def topk(s, k):
    p = np.zeros(len(s), int); p[np.argsort(-s)[:int(round(k))]] = 1; return p
def decide(rule, sv, sh, yv, yh):
    if rule == 'valthr': th = np.array([best_thr(yv[:, k], sv[:, k])[0] for k in range(6)]); return (sh >= th).astype(int)
    if rule.startswith('quota'): c = float(rule[5:]); return np.stack([topk(sh[:, k], c * yv[:, k].sum()) for k in range(6)], 1)
    if rule == 'oracle_k': return np.stack([topk(sh[:, k], yh[:, k].sum()) for k in range(6)], 1)
if __name__ == '__main__':
    runs = load()
    for val, seeds in runs.items():
        print('== val', val, 'seeds', sorted(seeds))
        for how in ('max', 'noisyor', 'mean'):
            for rule in ('valthr', 'quota0.7', 'quota0.85', 'quota1.0', 'oracle_k'):
                sc = []
                for s, z in seeds.items():
                    yv, yh = z['val_y'].astype(int), z['test_y'].astype(int)
                    sc.append(score(yh, decide(rule, agg(z['val_daily'], how), agg(z['test_daily'], how), yv, yh))[0])
                print(f'  {how:8s} {rule:10s} mean {np.mean(sc):.4f}  per-seed {np.round(sc, 4)}')
        # seed ensemble
        for how in ('max', 'noisyor'):
            for rule in ('valthr', 'quota0.85'):
                sv = np.mean([agg(z['val_daily'], how) for z in seeds.values()], 0); sh = np.mean([agg(z['test_daily'], how) for z in seeds.values()], 0)
                z = next(iter(seeds.values())); yv, yh = z['val_y'].astype(int), z['test_y'].astype(int)
                print(f'  ENSEMBLE {how:8s} {rule:10s} {score(yh, decide(rule, sv, sh, yv, yh))[0]:.4f}')
