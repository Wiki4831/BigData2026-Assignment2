import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np, pandas as pd
from tab import *
from sklearn.ensemble import HistGradientBoostingClassifier as H
cov, data, users, ev0 = load(); T = utc_timestamp
dense_end = T('2026-10-18')   # before this the source records push densely
def run(q, hold, val, last, seed=0):
    rng = np.random.default_rng(seed)
    ev = ev0
    if q < 1:
        drop = (ev.activity == 5) & (ev.event_time < dense_end) & (rng.random(len(ev)) > q)
        ev = ev[~drop]
    tr = pd.date_range(T('2026-10-04'), last, freq='D')
    X = np.vstack([features(ev, users, c, cov)[0] for c in tr]); Y = np.vstack([labels(ev, users, c) for c in tr])[:, 5]
    # evaluation uses the original (unthinned) data
    Xv, Xh = features(ev0, users, val, cov)[0], features(ev0, users, hold, cov)[0]
    Yv, Yh = labels(ev0, users, val)[:, 5], labels(ev0, users, hold)[:, 5]
    m = H(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=seed).fit(X, Y)
    sv, sh = m.predict_proba(Xv)[:, 1], m.predict_proba(Xh)[:, 1]
    th, _ = best_thr(Yv, sv); p = sh >= th
    tp = (p & (Yh == 1)).sum(); f1 = 2 * tp / (p.sum() + Yh.sum())
    from sklearn.metrics import average_precision_score as AP
    return f1, AP(Yh, sh), int(p.sum()), int(Yh.sum()), sh.sum(), Y.mean()
for hold, val, last in [(T('2026-11-08'), T('2026-11-01'), T('2026-10-25')), (T('2026-11-01'), T('2026-10-25'), T('2026-10-18'))]:
    for q in (1, 0.3, 0.1, 0.05):
        r = [run(q, hold, val, last, s) for s in (0, 1, 2)]
        print(hold.date(), f'q={q}: pushF1 {np.mean([x[0] for x in r]):.3f} AP {np.mean([x[1] for x in r]):.3f} pred {np.mean([x[2] for x in r]):.0f} (true {r[0][3]}) sum_p {np.mean([x[4] for x in r]):.0f} train_rate {np.mean([x[5] for x in r]):.3f}')
