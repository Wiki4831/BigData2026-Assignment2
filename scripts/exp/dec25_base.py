"""2025-12 baselines (GBDT, copy-last-week) and GNN summary, same protocol as the 2026 runs."""
import sys, glob; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier as H
from model_data import ActivityData, Coverage, utc_timestamp as T
from tab import features, labels, score, best_thr
from offline import agg
def five(Y, P):
    fs = []; TP = FP = FN = 0
    for i in range(5):
        tp = ((Y[:, i] == 1) & (P[:, i] == 1)).sum(); fp = ((Y[:, i] == 0) & (P[:, i] == 1)).sum(); fn = ((Y[:, i] == 1) & (P[:, i] == 0)).sum()
        fs.append(2 * tp / max(2 * tp + fp + fn, 1)); TP += tp; FP += fp; FN += fn
    return (np.mean(fs) + 2 * TP / max(2 * TP + FP + FN, 1)) / 2
cov = Coverage.read(Path('data/y2025dec/manifest.json'))
data = ActivityData(Path('data/y2025dec/processed/db'), cov)
users = pd.Index(pd.read_csv('scripts/cohort.txt', header=None)[0].astype('int64'))
ev = data.events.dropna(subset=['github_user_id']).copy()
ev['u'] = users.get_indexer(ev.github_user_id.astype('int64')); ev = ev[ev.u >= 0]
def build(cuts):
    X, Y = [], []
    for c in cuts:
        x, _ = features(ev, users, c, cov); X.append(x); Y.append(labels(ev, users, c))
    return np.vstack(X), np.vstack(Y)
D = 'data/model/relbench_public/err_study/'
for hold, val in (('2025-12-21', '2025-12-14'), ('2025-12-28', '2025-12-21')):
    h, v = T(hold), T(val)
    tr = pd.date_range(T('2025-12-02'), v - pd.Timedelta(days=7), freq='D')
    Xt, Yt = build(tr); Xv, Yv = build([v]); Xh, Yh = build([h])
    Sv = np.zeros(Yv.shape); Sh = np.zeros(Yh.shape)
    for k in range(6):
        m = H(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(Xt, Yt[:, k])
        Sv[:, k] = m.predict_proba(Xv)[:, 1]; Sh[:, k] = m.predict_proba(Xh)[:, 1]
    th = np.array([best_thr(Yv[:, k], Sv[:, k])[0] for k in range(6)]); P = (Sh >= th).astype(int)
    print(f'test {hold}: train days {len(tr)}  GBDT 6-class {score(Yh, P)[0]:.4f} 5-class {five(Yh, P):.4f}  pos {Yh.sum(0)}')
    print(f'test {hold}: copy-last-week 6-class {score(Yh, Yv)[0]:.4f} 5-class {five(Yh, Yv):.4f}')
    r6 = []; r5 = []
    for f in sorted(glob.glob(f'{D}d25_val{val}_s*.probs.npz')):
        z = np.load(f); yv, yh = z['val_y'].astype(int), z['test_y'].astype(int)
        sv, sh = agg(z['val_daily'], 'max'), agg(z['test_daily'], 'max')
        t = np.array([best_thr(yv[:, k], sv[:, k])[0] for k in range(6)]); Q = (sh >= t).astype(int)
        r6.append(score(yh, Q)[0]); r5.append(five(yh, Q))
    if r6: print(f'test {hold}: GNN n={len(r6)} 6-class {np.mean(r6):.4f} ({", ".join(f"{x:.3f}" for x in r6)}) 5-class {np.mean(r5):.4f}')
