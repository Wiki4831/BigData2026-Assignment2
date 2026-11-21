import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np, pandas as pd
from tab import *
from sklearn.ensemble import HistGradientBoostingClassifier as H
cov, data, users, ev = load()
T = utc_timestamp
cache = {}
def fl(c):
    if c not in cache: cache[c] = (features(ev, users, c, cov)[0], labels(ev, users, c))
    return cache[c]
def exp_f1_thr(p):
    # choose threshold maximizing expected F1 under calibrated p (plug-in): F ~ 2*sum(p over pred)/(npred+sum(p))
    o = np.argsort(-p); cs = np.cumsum(p[o]); k = np.arange(1, len(p) + 1); f = 2 * cs / (k + p.sum()); return p[o][int(np.argmax(f))]
def run(hold, val, last, half, name):
    tr = pd.date_range(T('2026-10-04'), last, freq='D')
    X = np.vstack([fl(c)[0] for c in tr]); Y = np.vstack([fl(c)[1] for c in tr])
    age = np.concatenate([np.full(len(users), (last - c).days) for c in tr])
    w = np.ones(len(age)) if half is None else 0.5 ** (age / half)
    Xv, Yv = fl(val); Xh, Yh = fl(hold)
    Sv = np.zeros(Yv.shape); Sh = np.zeros(Yh.shape)
    for k in range(6):
        m = H(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(X, Y[:, k], sample_weight=w)
        Sv[:, k] = m.predict_proba(Xv)[:, 1]; Sh[:, k] = m.predict_proba(Xh)[:, 1]
    res = {}
    thv = np.array([best_thr(Yv[:, k], Sv[:, k])[0] for k in range(6)])
    res['valthr'] = score(Yh, (Sh >= thv).astype(int))
    the = np.array([exp_f1_thr(Sh[:, k]) for k in range(6)])
    res['expF1(self-calib)'] = score(Yh, (Sh >= the).astype(int))
    tho = np.array([best_thr(Yh[:, k], Sh[:, k])[0] for k in range(6)])
    res['oracle'] = score(Yh, (Sh >= tho).astype(int))
    print(f'{name} half={half}', {k: round(v[0], 4) for k, v in res.items()}, 'expF1 per-act', np.round(res['expF1(self-calib)'][1], 3), 'predsum', np.round(Sh.sum(0)), 'pos', Yh.sum(0))
for half in (None, 14, 7, 4):
    run(T('2026-11-08'), T('2026-11-01'), T('2026-10-25'), half, 'H1108')
    run(T('2026-11-01'), T('2026-10-25'), T('2026-10-18'), half, 'H1101')
