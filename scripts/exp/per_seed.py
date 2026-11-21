import sys, glob; sys.path.insert(0,'scripts'); sys.path.insert(0,'scripts/exp')
import numpy as np
from tab import score, best_thr
from offline import agg
D='data/model/relbench_public/err_study/'
for tag in sys.argv[1:]:
  for val in ('2026-10-25','2026-11-01'):
    r=[]; ap=[]
    for f in sorted(glob.glob(f'{D}{tag}_val{val}_s*.probs.npz')):
        z=np.load(f);yv,yh=z['val_y'].astype(int),z['test_y'].astype(int);sv,sh=agg(z['val_daily'],'max'),agg(z['test_daily'],'max')
        th=np.array([best_thr(yv[:,k],sv[:,k])[0] for k in range(6)]);r.append(score(yh,(sh>=th).astype(int))[0])
    print(f'{tag:7s} val {val}: '+' '.join(f'{x*100:.2f}' for x in r)+f'  mean {np.mean(r)*100:.2f}')
