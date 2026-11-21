import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); p = d.pivot_table(index=['hour', 'events'], columns='type', values='count', fill_value=0).reset_index()
tcols = ['PushEvent', 'PullRequestEvent', 'IssuesEvent', 'IssueCommentEvent', 'PullRequestReviewEvent', 'PullRequestReviewCommentEvent', 'WatchEvent', 'CreateEvent']
for c in tcols:
    if c not in p: p[c] = 0
p['push%'] = 100 * p.PushEvent / p.events
p['other5/1000'] = 1000 * (p.PullRequestEvent + p.IssuesEvent + p.IssueCommentEvent + p.PullRequestReviewEvent + p.PullRequestReviewCommentEvent) / p.events
p['watch%'] = 100 * p.WatchEvent / p.events
pd.set_option('display.width', 200)
out = p[['hour', 'events', 'PushEvent', 'push%', 'PullRequestEvent', 'IssueCommentEvent', 'other5/1000', 'watch%']].sort_values('hour')
print(out.round(1).to_string(index=False))
