#!/bin/bash
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
xargs -P 6 -n 1 scripts/exp/dl_hour.sh < data/y2025/hours.txt > data/y2025/download.log 2>&1
/home/asiadragon/miniconda3/envs/bigdata/bin/python - <<'P'
import json, datetime as d, os
files=[]; t=d.datetime(2025,11,1)
while t<d.datetime(2025,12,1):
    n=f'{t:%Y-%m-%d}-{t.hour}.json.gz'; ok=os.path.exists(f'data/y2025/raw/{n}')
    files.append({'hour_utc':t.isoformat()+'+00:00','file':n,'status':'downloaded' if ok else 'missing'}); t+=d.timedelta(hours=1)
json.dump({'start_utc_inclusive':'2025-11-01T00:00:00+00:00','end_utc_exclusive':'2025-12-01T00:00:00+00:00','files':files},open('data/y2025/download_manifest.json','w'))
print(sum(f['status']=='downloaded' for f in files),'of',len(files))
P
/home/asiadragon/miniconda3/envs/bigdata/bin/python -u scripts/preprocess.py --raw data/y2025/raw --out data/y2025/processed > data/y2025/preprocess.log 2>&1
echo finished > data/y2025/done.flag
