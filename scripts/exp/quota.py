import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np, pandas as pd
from tab import *
from sklearn.ensemble import HistGradientBoostingClassifier as H
cov, data, users, ev = load(); T = utc_timestamp
cache = {}
def fl(c):
    if c not in cache: cache[c] = (features(ev, users, c, cov)[0], labels(ev, users, c))
    return cache[c]
def topk(s, k):
    p = np.zeros(len(s), int); p[np.argsort(-s)[:int(round(k))]] = 1; return p
def gbdt(hold, val, last):
    tr = pd.date_range(T('2026-10-04'), last, freq='D')
    X = np.vstack([fl(c)[0] for c in tr]); Y = np.vstack([fl(c)[1] for c in tr])
    Sh = np.zeros((len(users), 6)); Sv = Sh.copy()
    for k in range(6):
        m = H(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(X, Y[:, k])
        Sh[:, k] = m.predict_proba(fl(hold)[0])[:, 1]; Sv[:, k] = m.predict_proba(fl(val)[0])[:, 1]
    return Sv, Sh
def report(tag, Sv, Sh, hold, val):
    Yv, Yh = fl(val)[1], fl(hold)[1]
    thv = np.array([best_thr(Yv[:, k], Sv[:, k])[0] for k in range(6)])
    print(tag, 'valthr %.4f' % score(Yh, (Sh >= thv).astype(int))[0], end=' | ')
    for c in (0.7, 0.85, 1.0, 1.2):
        P = np.stack([topk(Sh[:, k], c * Yv[:, k].sum()) for k in range(6)], 1)
        print(f'quota x{c}: %.4f' % score(Yh, P)[0], end='  ')
    # oracle-k (true count)
    P = np.stack([topk(Sh[:, k], Yh[:, k].sum()) for k in range(6)], 1); print('| k=truecount %.4f' % score(Yh, P)[0])
for hold, val, last in [(T('2026-11-08'), T('2026-11-01'), T('2026-10-25')), (T('2026-11-01'), T('2026-10-25'), T('2026-10-18'))]:
    Sv, Sh = gbdt(hold, val, last); report('GBDT ' + str(hold.date()), Sv, Sh, hold, val)
# GNN holdout scores for 11/8 (val-week scores not saved -> use quota only)
d = pd.read_pickle('/tmp/claude-1000/-home-asiadragon-geolife/45e26f38-921d-4dcf-9d54-e2a928565e09/scratchpad/d.pkl')
d = d.set_index('github_user_id').reindex(users)
acts = EVENT_COLS
Sg = np.stack([d[a + '_max_daily_probability'].to_numpy() for a in acts], 1); Yh = fl(T('2026-11-08'))[1]; Yv = fl(T('2026-11-01'))[1]
assert (np.stack([d[a + '_actual'] for a in acts], 1) == Yh).mean() > 0.999, (np.stack([d[a + '_actual'] for a in acts], 1) == Yh).mean()
print('GNN current', score(Yh, np.stack([d[a + '_prediction'].to_numpy() for a in acts], 1))[0])
for c in (0.7, 0.85, 1.0, 1.2):
    print('GNN quota x', c, round(score(Yh, np.stack([topk(Sg[:, k], c * Yv[:, k].sum()) for k in range(6)], 1))[0], 4))
print('GNN k=truecount', round(score(Yh, np.stack([topk(Sg[:, k], Yh[:, k].sum()) for k in range(6)], 1))[0], 4))

print('=== ensemble 11/8')
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
Sv8, Sh8 = gbdt(T('2026-11-08'), T('2026-11-01'), T('2026-10-25'))
hist = d.has_history.to_numpy()
from scipy.stats import rankdata
def r(s): return rankdata(s) / len(s)
for nm, S in [('GBDT', Sh8), ('GNN', Sg), ('rankavg', np.stack([(r(Sh8[:, k]) + r(Sg[:, k])) / 2 for k in range(6)], 1)),
              ('rank 0.3G+0.7N', np.stack([(0.3 * r(Sh8[:, k]) + 0.7 * r(Sg[:, k])) for k in range(6)], 1))]:
    ap = np.mean([AP(Yh[hist, k], S[hist, k]) for k in range(6)]); au = np.mean([AUC(Yh[hist, k], S[hist, k]) for k in range(6)])
    print(nm, 'AP(hist) %.3f AUC %.3f' % (ap, au), 'quota.85 %.4f' % score(Yh, np.stack([topk(S[:, k], 0.85 * Yv[:, k].sum()) for k in range(6)], 1))[0],
          'k=true %.4f' % score(Yh, np.stack([topk(S[:, k], Yh[:, k].sum()) for k in range(6)], 1))[0])
