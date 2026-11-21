"""Generate variant comparison tables (mean±std over seeds) for 2026 and 2025-12 runs."""
import sys, glob, re; sys.path.insert(0,'scripts'); sys.path.insert(0,'scripts/exp')
import numpy as np
from tab import score, best_thr
from offline import agg
D='data/model/relbench_public/err_study/'
def five(Y,P):
    fs=[];TP=FP=FN=0
    for i in range(5):
        tp=((Y[:,i]==1)&(P[:,i]==1)).sum();fp=((Y[:,i]==0)&(P[:,i]==1)).sum();fn=((Y[:,i]==1)&(P[:,i]==0)).sum()
        fs.append(2*tp/max(2*tp+fp+fn,1));TP+=tp;FP+=fp;FN+=fn
    return (np.mean(fs)+2*TP/max(2*TP+FP+FN,1))/2
def run(tag,val):
    r=[]
    for f in sorted(glob.glob(f'{D}{tag}_val{val}_s*.probs.npz')):
        z=np.load(f);yv,yh=z['val_y'].astype(int),z['test_y'].astype(int)
        sv,sh=(z['val'],z['test']) if z['val_daily'].ndim==0 else (agg(z['val_daily'],'max'),agg(z['test_daily'],'max'))
        th=np.array([best_thr(yv[:,k],sv[:,k])[0] for k in range(6)]);P=(sh>=th).astype(int)
        r.append((score(yh,P)[0],five(yh,P)))
    return np.array(r)
def cell(a,j): return f'{100*a[:,j].mean():.1f}±{100*a[:,j].std():.1f}' if len(a) else '-'
def table(title,tags,vals,names,base):
    print(f'\n### {title}\n\n| 變體 | 說明 | '+' | '.join(f'測 {t}（5類）' for _,t in vals)+' | 平均5類 | 平均6類 | 對基準 |\n|---|---|'+'---|'*(len(vals)+3))
    bm=np.mean([run(base,v)[:,1].mean() for v,_ in vals])
    for tag in tags:
        rs=[run(tag,v) for v,_ in vals]
        if not all(len(r) for r in rs): continue
        m5=np.mean([r[:,1].mean() for r in rs]);m6=np.mean([r[:,0].mean() for r in rs])
        print(f'| {tag} | {names.get(tag,"")} | '+' | '.join(cell(r,1) for r in rs)+f' | {100*m5:.1f} | {100*m6:.1f} | {100*(m5-bm):+.1f} |')
names={'h3':'基準：3 hop SAGE＋預訓練＋GRU','hstats':'+使用者統計特徵（預訓練後加入）','hpre':'統計特徵放進預訓練','hseq':'+最近32筆事件序列','hboth':'+統計＋序列','hv2':'統計 v2（小時、週末、間隔、覆蓋率）',
'edge':'+PR→Repo、Repo→Owner 額外邊','grp':'+owner/org 群組節點','tlast':'時間取樣取最新鄰居','rg3':'RelGNN 骨幹','notime':'關掉相對時間編碼','hop2':'2 hop','nopre':'不預訓練','weekly':'週標籤頭（不用逐日）','ft':'編碼器微調','norec':'拿掉 GRU 狀態更新（重用基準編碼器）','seqe':'先後邊（同使用者同表的前一筆事件）','focal':'focal loss γ=2','augde':'DropEdge 0.2（預訓練）','augem':'遮蔽使用者自身事件邊 0.3（預訓練）','augde1g':'DropEdge 0.1（預訓練＋GRU）','gap':'訓練對齊資料缺口','rank05':'排序損失 0.5'}
t26=['h3','hstats','hpre','hseq','hboth','hv2','edge','grp','tlast','rg3','notime','hop2','nopre','weekly','ft','gap','rank05','augde','augem','augde1g','seqe','focal','norec']
table('2026：測 11/1、11/8（驗證 10/25、11/1，3 seeds）',t26,[('2026-10-25','11/1'),('2026-11-01','11/8')],names,'h3')
d25={('d25'+k[1:] if k!='h3' else 'd25'):v for k,v in names.items()}
d25n={'d25':'基準'}|{'d25'+k:v for k,v in names.items() if k!='h3'}
table('2025-12：測 12/21（驗證 12/14，3 seeds）',['d25']+['d25'+k for k in ['hstats','hv2','hseq','hboth','ft','edge','tlast','notime','augde','augem','seqe','focal','norec']],[('2025-12-14','12/21')],d25n,'d25')
