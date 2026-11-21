import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC
d = pd.read_pickle('/tmp/claude-1000/-home-asiadragon-geolife/45e26f38-921d-4dcf-9d54-e2a928565e09/scratchpad/d.pkl'); d = d[d.has_history]
for a in ['open_pr','open_issue','issue_comment','pr_review','pr_review_comment','push']:
    y=d[a+'_actual'];s=d[a+'_max_daily_probability']
    g={'lastweek':d[a+'_lw'],'earlier(not lw)':d[a+'_ever']&~d[a+'_lw'],'never':~d[a+'_ever']}
    out=[]
    for n,m in g.items():
        out.append(f'{n}: n={m.sum()} rate={y[m].mean():.3f} auc={AUC(y[m],s[m]) if 0<y[m].sum()<m.sum() else float("nan"):.3f} pos={y[m].sum()} meanp={s[m].mean():.3f}')
    print(a,'\n   '+'\n   '.join(out))
