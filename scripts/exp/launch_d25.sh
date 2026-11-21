#!/bin/bash
# usage: launch_d25.sh TAG "extra args" [reuse|full]   2025-12 data, test week 12/21 (val 12/14)
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
PY=data/model/relbench_public/venv/bin/python
tag=$1; EXTRA="$2"; MODE=${3:-reuse}
for seed in 42 57 2026; do
  out=data/model/relbench_public/err_study/d25${tag}_val2025-12-14_s$seed
  [ -f $out.json ] && continue
  if [ $MODE = reuse ]; then CK="--encoder-checkpoint data/model/relbench_public/err_study/d25_val2025-12-14_s$seed.encoder.pt"; else CK="--pretrain-epochs 3"; fi
  $PY scripts/relbench_asg2.py --device cuda --hops 3 --daily-loss $CK --seed $seed --clip-old-days 365 \
    --db data/y2025dec/processed/db --manifest data/y2025dec/manifest.json --source-gap-csv data/y2025dec/no_gaps.csv \
    --validation-cutoff 2025-12-14 --holdout-cutoff 2025-12-21 --metrics $out.json --save-probabilities $EXTRA > $out.log 2>&1
done
