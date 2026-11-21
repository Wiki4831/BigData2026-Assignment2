import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np, pandas as pd
from tab import *
from sklearn.ensemble import HistGradientBoostingClassifier as H
cov, data, users, ev = load(); T = utc_timestamp
cache = {}
def fl(c):
    if c not in cache: cache[c] = (features(ev, users, c, cov)[0], labels(ev, users, c))
    return cache[c]
def sc5(Y, P): return score(Y[:, :5], P[:, :5])[0] if False else None
def five(Y, P):
    fs = []; TP = FP = FN = 0
    for i in range(5):
        tp = ((Y[:, i] == 1) & (P[:, i] == 1)).sum(); fp = ((Y[:, i] == 0) & (P[:, i] == 1)).sum(); fn = ((Y[:, i] == 1) & (P[:, i] == 0)).sum()
        fs.append(2 * tp / max(2 * tp + fp + fn, 1)); TP += tp; FP += fp; FN += fn
    return (np.mean(fs) + 2 * TP / max(2 * TP + FP + FN, 1)) / 2
print('test week | train cutoffs | copy-last-week(6) | GBDT valthr (6) | GBDT 5-class | per-act F1')
for hold in ['2026-10-25', '2026-11-01', '2026-11-08']:
    h = T(hold); v = h - pd.Timedelta(days=7); last = v - pd.Timedelta(days=7)
    tr = pd.date_range(T('2026-10-04'), last, freq='D')
    X = np.vstack([fl(c)[0] for c in tr]); Y = np.vstack([fl(c)[1] for c in tr])
    Xv, Yv = fl(v); Xh, Yh = fl(h)
    # copy last week = activity in the 7 days before cutoff
    prev = fl(h - pd.Timedelta(days=7))[1]
    r = []
    for seed in (0, 1, 2):
        Sv = np.zeros(Yv.shape); Sh = Sv.copy()
        for k in range(6):
            m = H(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=seed).fit(X, Y[:, k])
            Sv[:, k] = m.predict_proba(Xv)[:, 1]; Sh[:, k] = m.predict_proba(Xh)[:, 1]
        th = np.array([best_thr(Yv[:, k], Sv[:, k])[0] for k in range(6)]); P = (Sh >= th).astype(int)
        r.append((score(Yh, P)[0], five(Yh, P), score(Yh, P)[1]))
    print(hold, len(tr), '| %.4f' % score(Yh, prev)[0], '| %.4f' % np.mean([x[0] for x in r]), '| %.4f' % np.mean([x[1] for x in r]), '|', np.round(np.mean([x[2] for x in r], 0), 3), '| copy5 %.4f' % five(Yh, prev))
