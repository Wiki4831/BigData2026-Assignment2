import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np, pandas as pd
from tab import *
from sklearn.ensemble import HistGradientBoostingClassifier as H
cov, data, users, ev = load(); T = utc_timestamp
names = features(ev, users, T('2026-10-10'), cov)[1]
cache = {}
def fl(c):
    if c not in cache: cache[c] = (features(ev, users, c, cov)[0], labels(ev, users, c))
    return cache[c]
ci = [names.index(f'{a}_c28') for a in EVENT_COLS]
tot = lambda X: np.expm1(X[:, ci]).sum(1)          # events in last 28d
rules = {'all': lambda X: np.ones(len(X), bool),
         'any28>=1': lambda X: tot(X) >= 1,
         'any28>=2': lambda X: tot(X) >= 2,
         'any28>=4': lambda X: tot(X) >= 4,
         'any28>=1 + 20% of rest': None}
rng = np.random.default_rng(0)
for hold, val, last in [(T('2026-11-08'), T('2026-11-01'), T('2026-10-25')), (T('2026-11-01'), T('2026-10-25'), T('2026-10-18'))]:
    tr = pd.date_range(T('2026-10-04'), last, freq='D')
    X = np.vstack([fl(c)[0] for c in tr]); Y = np.vstack([fl(c)[1] for c in tr])
    Xv, Yv = fl(val); Xh, Yh = fl(hold)
    for nm, r in rules.items():
        keep = (tot(X) >= 1) | (rng.random(len(X)) < 0.2) if r is None else r(X)
        out = []
        for seed in (0, 1, 2):
            Sv = np.zeros(Yv.shape); Sh = Sv.copy()
            for k in range(6):
                m = H(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=seed).fit(X[keep], Y[keep, k])
                Sv[:, k] = m.predict_proba(Xv)[:, 1]; Sh[:, k] = m.predict_proba(Xh)[:, 1]
            th = np.array([best_thr(Yv[:, k], Sv[:, k])[0] for k in range(6)])
            out.append(score(Yh, (Sh >= th).astype(int))[0])
        # share of holdout positives among users the rule would drop
        dropped = ~(tot(Xh) >= 1) if nm != 'all' else np.zeros(len(Xh), bool)
        print(hold.date(), f'{nm:24s} kept {keep.mean():.2f}  score {np.mean(out):.4f}  (seeds {np.round(out, 4)})  holdout pos among zero-28d users: {int(Yh[dropped].sum()) if dropped.any() else "-"}/{int(Yh.sum())}')
