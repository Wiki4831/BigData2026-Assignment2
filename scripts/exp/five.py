import sys, glob; sys.path.insert(0,'scripts'); sys.path.insert(0,'scripts/exp')
import numpy as np
from tab import score, best_thr
from offline import agg
def five(Y,P):
    fs=[];TP=FP=FN=0
    for i in range(5):
        tp=((Y[:,i]==1)&(P[:,i]==1)).sum();fp=((Y[:,i]==0)&(P[:,i]==1)).sum();fn=((Y[:,i]==1)&(P[:,i]==0)).sum()
        fs.append(2*tp/max(2*tp+fp+fn,1));TP+=tp;FP+=fp;FN+=fn
    return (np.mean(fs)+2*TP/max(2*TP+FP+FN,1))/2
D='data/model/relbench_public/err_study/'
for tag in sys.argv[1:]:
  for val,test in [('2026-10-18','10/25'),('2026-10-25','11/1'),('2026-11-01','11/8')]:
    r6=[];r5=[]
    for f in sorted(glob.glob(f'{D}{tag}_val{val}_s*.probs.npz')):
        z=np.load(f);yv,yh=z['val_y'].astype(int),z['test_y'].astype(int);sv,sh=agg(z['val_daily'],'max'),agg(z['test_daily'],'max')
        th=np.array([best_thr(yv[:,k],sv[:,k])[0] for k in range(6)]);P=(sh>=th).astype(int)
        r6.append(score(yh,P)[0]);r5.append(five(yh,P))
    if r6: print(f'{tag:7s} test {test}: n={len(r6)} 6-class {np.mean(r6):.4f}  5-class {np.mean(r5):.4f}')
