import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'scripts/exp')
import numpy as np, pandas as pd
from tab import *
cov,data,users,ev=load(); T=utc_timestamp
ws=list(pd.date_range(T('2026-10-04'),T('2026-11-08'),freq='7D'))
L={w:labels(ev,users,w) for w in ws}
print('persistence P(active week t+1 | active week t) and prevalence, by activity')
print('week t -> t+1   '+'  '.join(f'{a[:10]:>10s}' for a in EVENT_COLS))
for a,b in zip(ws[:-1],ws[1:]):
    p=[(L[a][:,k]*L[b][:,k]).sum()/max(L[a][:,k].sum(),1) for k in range(6)]
    print(f'{(a+pd.Timedelta(hours=8)).date()}  P(b|a) '+'  '.join(f'{x:10.3f}' for x in p))
print('prevalence(%) '+'  '.join(f'{a[:10]:>10s}' for a in EVENT_COLS))
for w in ws: print(f'{(w+pd.Timedelta(hours=8)).date()}  '+'  '.join(f'{100*L[w][:,k].mean():10.2f}' for k in range(6)))
# cohort decay: users active (any non-push) in week 1 vs never
nonpush=lambda w: L[w][:,:5].max(1)
act1=nonpush(ws[0])==1
print('\nany non-push activity rate: users active in wk1 vs not')
for w in ws: print((w+pd.Timedelta(hours=8)).date(), f'active-in-wk1: {nonpush(w)[act1].mean():.3f}  others: {nonpush(w)[~act1].mean():.3f}')
