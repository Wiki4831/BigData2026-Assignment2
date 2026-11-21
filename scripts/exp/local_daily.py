import sys, pandas as pd, numpy as np
d = pd.read_csv(sys.argv[1]); p = d.pivot_table(index=['hour', 'events'], columns='type', values='count', fill_value=0).reset_index()
p['t'] = pd.to_datetime(p.hour.str.rsplit('-', n=1).str[0]) + pd.to_timedelta(p.hour.str.rsplit('-', n=1).str[1].astype(int), unit='h')
p['day'] = p.t.dt.floor('D')
five = ['PullRequestEvent', 'IssuesEvent', 'IssueCommentEvent', 'PullRequestReviewEvent', 'PullRequestReviewCommentEvent']
for c in five + ['PushEvent']:
    if c not in p: p[c] = 0
p['five'] = p[five].sum(1)
p['pushshare'] = p.PushEvent / p.events; p['five1000'] = 1000 * p.five / p.events
g = p.groupby('day').agg(hours=('hour', 'size'), events=('events', 'sum'), push=('PushEvent', 'sum'), five=('five', 'sum'))
g['events/hr k'] = g.events / g.hours / 1000; g['push%'] = 100 * g.push / g.events; g['five/1000'] = 1000 * g.five / g.events
g['hrs push<20%'] = p[p.pushshare < .2].groupby('day').size().reindex(g.index, fill_value=0)
g['hrs five<60/1000'] = p[p.five1000 < 60].groupby('day').size().reindex(g.index, fill_value=0)
pd.set_option('display.width', 220)
print(g[['events/hr k', 'push%', 'five/1000', 'hrs push<20%', 'hrs five<60/1000']].round(1).to_string())
print('\nhours', len(p), ' push<20%:', int((p.pushshare < .2).sum()), ' five<60/1000:', int((p.five1000 < 60).sum()), ' both normal:', int(((p.pushshare >= .2) & (p.five1000 >= 60)).sum()))
