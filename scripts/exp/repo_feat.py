import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/exp')
import numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from model_data import EVENT_COLS
SRC = {'open_pr': ('PullRequests', 'author_id', 'created_at', 'created_at'),
       'open_issue': ('Issues', 'author_id', 'created_at', 'first_seen_at'),
       'issue_comment': ('IssueComments', 'author_id', 'created_at', 'created_at'),
       'pr_review': ('PRReviews', 'reviewer_id', 'submitted_at', 'first_seen_at'),
       'pr_review_comment': ('PRReviewComments', 'author_id', 'created_at', 'created_at'),
       'push': ('Pushes', 'user_id', 'created_at', 'created_at')}
def repo_events(users):
    """all events (any user) with repo, plus cohort index of actor (-1 if not cohort)"""
    U = pd.read_parquet('data/processed/db/Users.parquet', columns=['user_id', 'github_user_id']).set_index('user_id').github_user_id
    out = []
    for k, a in enumerate(EVENT_COLS):
        t, actor, et, rt = SRC[a]
        cols = ['repo_id', actor, et] + ([rt] if rt != et else [])
        if a == 'open_issue': cols += ['is_pull_request']
        x = pd.read_parquet(f'data/processed/db/{t}.parquet', columns=list(dict.fromkeys(cols)))
        if a == 'open_issue': x = x[x.is_pull_request.eq(False).fillna(False)]
        e = pd.DataFrame({'repo': x.repo_id.to_numpy(), 'actor': x[actor].to_numpy(), 'event_time': x[et].to_numpy(), 'row_time': x[rt].to_numpy(), 'activity': k})
        out.append(e)
    e = pd.concat(out, ignore_index=True).dropna(subset=['repo', 'actor', 'event_time', 'row_time'])
    e['gh'] = e.actor.map(U); e['u'] = users.get_indexer(e.gh.fillna(-1).astype('int64')); e['u'] = e.u.where(e.gh.notna(), -1)
    e['repo'] = e.repo.astype('int64'); rid = {r: i for i, r in enumerate(np.unique(e.repo))}; e['r'] = e.repo.map(rid)
    prs = pd.read_parquet('data/processed/db/PullRequests.parquet', columns=['pr_id', 'repo_id', 'author_id', 'created_at', 'first_seen_at']).dropna(subset=['repo_id', 'created_at'])
    prs['r'] = prs.repo_id.astype('int64').map(rid)
    rv = pd.read_parquet('data/processed/db/PRReviews.parquet', columns=['pr_id', 'submitted_at', 'first_seen_at'])
    return e, len(rid), prs, rv

def repo_features(E, nrepo, prs, rv, users, cut):
    e = E[(E.row_time < cut) & (E.event_time < cut)]
    age = (cut - e.event_time).dt.total_seconds().to_numpy() / 86400
    n = len(users); cols = []
    mine = e[(e.u >= 0) & (age <= 28)]
    M = sp.csr_matrix((np.ones(len(mine)), (mine.u.to_numpy(), mine.r.to_numpy())), shape=(n, nrepo)); M.data[:] = 1; M.sum_duplicates(); M.data[:] = 1
    cols.append(np.asarray(M.sum(1)).ravel())  # n repos
    heats = []
    for d in (3, 7):
        for k in range(6):
            m = (e.activity.to_numpy() == k) & (age <= d)
            heats.append(np.bincount(e.r.to_numpy()[m], minlength=nrepo).astype(float))
    # pending PRs: opened within 14d, no review seen before cut
    p = prs[(prs.first_seen_at < cut) & (prs.created_at < cut) & (prs.created_at >= cut - pd.Timedelta(days=14))]
    rvc = set(rv[(rv.submitted_at < cut) & (rv.first_seen_at < cut)].pr_id)
    p = p[~p.pr_id.isin(rvc)]
    pend = np.bincount(p.r.dropna().astype(int), minlength=nrepo).astype(float)
    pend_other = pend  # author excluded below per user via simple approx (own PRs count too)
    heats.append(pend)
    # distinct other cohort users per repo
    cu = e[(e.u >= 0) & (age <= 28)][['u', 'r']].drop_duplicates()
    heats.append(np.bincount(cu.r.to_numpy(), minlength=nrepo).astype(float))
    for H in heats:
        cols.append(np.log1p(M @ H))
        mx = M.multiply(H[None, :]).tocsr().max(axis=1).toarray().ravel(); cols.append(np.log1p(mx))
    return np.column_stack(cols).astype('float32')
