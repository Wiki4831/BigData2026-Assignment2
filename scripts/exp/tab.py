"""Tabular GBDT baseline on per-user event history, same protocol as the GNN reports.
Train on daily cutoffs, pick thresholds on week 11/1, score holdout 11/8."""
import sys; sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from pathlib import Path
from model_data import ActivityData, Coverage, WEEK, DAY, utc_timestamp, EVENT_COLS

WIN = (1, 3, 7, 14, 28)

def load():
    cov = Coverage.read(Path('data/download_manifest.json'))
    data = ActivityData(Path('data/processed/db'), cov)
    users = pd.Index(pd.read_csv('scripts/cohort.txt', header=None)[0].astype('int64'))
    ev = data.events.dropna(subset=['github_user_id']).copy()
    ev['u'] = users.get_indexer(ev.github_user_id.astype('int64'))
    ev = ev[ev.u >= 0]
    return cov, data, users, ev

def features(ev, users, cut, cov):
    n = len(users)
    prior = ev[(ev.row_time < cut) & (ev.event_time < cut)]
    cols, names = [], []
    for k, a in enumerate(EVENT_COLS):
        pk = prior[prior.activity == k]
        age = (cut - pk.event_time).dt.total_seconds().to_numpy() / 86400
        u = pk.u.to_numpy()
        for w in WIN:
            c = np.bincount(u[age <= w], minlength=n); cols.append(np.log1p(c)); names.append(f'{a}_c{w}')
        last = np.full(n, np.nan)
        np.fmin.at(last, u, age); cols.append(np.log1p(last)); names.append(f'{a}_since')
        # active days in last 28d
        day = np.floor(age).astype(int); m = day < 28
        ad = pd.DataFrame({'u': u[m], 'd': day[m]}).drop_duplicates().groupby('u').size()
        cols.append(ad.reindex(range(n), fill_value=0).to_numpy()); names.append(f'{a}_ad28')
    allp = prior; age = (cut - allp.event_time).dt.total_seconds().to_numpy() / 86400
    day = np.floor(age).astype(int); m = day < 28
    ad = pd.DataFrame({'u': allp.u.to_numpy()[m], 'd': day[m]}).drop_duplicates().groupby('u').size()
    cols.append(ad.reindex(range(n), fill_value=0).to_numpy()); names.append('any_ad28')
    cols.append(np.log1p(np.bincount(allp.u, minlength=n))); names.append('any_total')
    hist = (cut - ev.event_time.min()).total_seconds() / 86400
    cols.append(np.full(n, hist)); names.append('hist_days')
    cols.append(np.full(n, pd.Timestamp(cut + pd.Timedelta(hours=8)).dayofweek)); names.append('dow')
    return np.column_stack(cols).astype('float32'), names

def labels(ev, users, cut, horizon=WEEK):
    end = cut + horizon
    f = ev[(ev.event_time >= cut) & (ev.event_time < end) & (ev.row_time < end)]
    Y = np.zeros((len(users), 6), dtype=np.int8)
    for k in range(6):
        Y[np.unique(f[f.activity == k].u), k] = 1
    return Y

def score(Y, P):
    fs = []; TP = FP = FN = 0
    for i in range(6):
        tp = ((Y[:, i] == 1) & (P[:, i] == 1)).sum(); fp = ((Y[:, i] == 0) & (P[:, i] == 1)).sum(); fn = ((Y[:, i] == 1) & (P[:, i] == 0)).sum()
        fs.append(2 * tp / max(2 * tp + fp + fn, 1)); TP += tp; FP += fp; FN += fn
    return (np.mean(fs) + 2 * TP / max(2 * TP + FP + FN, 1)) / 2, fs

def best_thr(y, s):
    o = np.argsort(-s); ys = y[o]; tp = np.cumsum(ys); k = np.arange(1, len(s) + 1)
    f = 2 * tp / (k + y.sum()); i = int(np.argmax(f)); return s[o][i], f[i]
