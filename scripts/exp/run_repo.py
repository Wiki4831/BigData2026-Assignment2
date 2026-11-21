import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np, pandas as pd, time
from tab import *; from repo_feat import *
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
cov, data, users, ev = load(); T = utc_timestamp
t0 = time.time(); E, nrepo, prs, rv = repo_events(users); print('repo events', len(E), nrepo, time.time() - t0)
cache = {}
def fl(c):
    if c not in cache:
        x = features(ev, users, c, cov)[0]; r = repo_feat = repo_features(E, nrepo, prs, rv, users, c)
        cache[c] = (x, np.hstack([x, r]), labels(ev, users, c))
    return cache[c]
def topk(s, k):
    p = np.zeros(len(s), int); p[np.argsort(-s)[:int(round(k))]] = 1; return p
for hold, val, last in [(T('2026-11-08'), T('2026-11-01'), T('2026-10-25')), (T('2026-11-01'), T('2026-10-25'), T('2026-10-18'))]:
    tr = pd.date_range(T('2026-10-04'), last, freq='D')
    Yh = fl(hold)[2]; Yv = fl(val)[2]
    for name, idx in [('base', 0), ('+repo', 1)]:
        X = np.vstack([fl(c)[idx] for c in tr]); Y = np.vstack([fl(c)[2] for c in tr])
        Sh = np.zeros((len(users), 6)); Sv = Sh.copy()
        for k in range(6):
            m = H(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(X, Y[:, k])
            Sh[:, k] = m.predict_proba(fl(hold)[idx])[:, 1]; Sv[:, k] = m.predict_proba(fl(val)[idx])[:, 1]
        thv = np.array([best_thr(Yv[:, k], Sv[:, k])[0] for k in range(6)])
        print(hold.date(), name, 'AP', np.round([AP(Yh[:, k], Sh[:, k]) for k in range(6)], 3), 'meanAP %.3f' % np.mean([AP(Yh[:, k], Sh[:, k]) for k in range(6)]),
              'valthr %.4f' % score(Yh, (Sh >= thv).astype(int))[0], 'k=true %.4f' % score(Yh, np.stack([topk(Sh[:, k], Yh[:, k].sum()) for k in range(6)], 1))[0])
