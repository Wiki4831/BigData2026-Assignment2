import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np, pandas as pd, pickle
from tab import *
from sklearn.ensemble import HistGradientBoostingClassifier as H
cov, data, users, ev = load()
T = lambda s: utc_timestamp(s)
def build(cuts):
    X, Y, W = [], [], []
    for c in cuts:
        x, names = features(ev, users, c, cov); X.append(x); Y.append(labels(ev, users, c)); W.append(np.full(len(users), (c - T('2026-10-04')).days))
    return np.vstack(X), np.vstack(Y), np.concatenate(W), names
def run(hold, val, last_train_cut):
    tr = pd.date_range(T('2026-10-04'), last_train_cut, freq='D')
    Xt, Yt, Wt, names = build(tr)
    Xv, Yv, _, _ = build([val]); Xh, Yh, _, _ = build([hold])
    Sv = np.zeros(Yv.shape); Sh = np.zeros(Yh.shape)
    for k in range(6):
        m = H(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(Xt, Yt[:, k])
        Sv[:, k] = m.predict_proba(Xv)[:, 1]; Sh[:, k] = m.predict_proba(Xh)[:, 1]
    th = [best_thr(Yv[:, k], Sv[:, k])[0] for k in range(6)]
    P = (Sh >= np.array(th)).astype(int)
    sc, fs = score(Yh, P)
    print('hold', hold, 'score %.4f' % sc, np.round(fs, 3), 'pred', P.sum(0), 'pos', Yh.sum(0))
    print(' val-week score(optimistic) %.4f' % score(Yv, (Sv >= np.array(th)).astype(int))[0], 'val pos', Yv.sum(0))
    return Sv, Sh, Yv, Yh
if __name__ == '__main__':
    out = {}
    out['h1108'] = run(T('2026-11-08'), T('2026-11-01'), T('2026-10-25'))
    out['h1101'] = run(T('2026-11-01'), T('2026-10-25'), T('2026-10-18'))
    pickle.dump(out, open('/tmp/claude-1000/-home-asiadragon-geolife/45e26f38-921d-4dcf-9d54-e2a928565e09/scratchpad/tab_out.pkl', 'wb'))
