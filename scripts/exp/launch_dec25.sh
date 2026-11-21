#!/bin/bash
# same method on 2025-12: test weeks 12/21 (val 12/14) and 12/28 (val 12/21, partial week)
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
PY=data/model/relbench_public/venv/bin/python
for seed in 42 57 2026; do
  for pair in "2025-12-14 2025-12-21" "2025-12-21 2025-12-28"; do
    set -- $pair
    out=data/model/relbench_public/err_study/d25_val$1_s$seed
    [ -f $out.json ] && continue
    $PY scripts/relbench_asg2.py --device cuda --hops 3 --daily-loss --pretrain-epochs 3 --seed $seed --clip-old-days 365 \
      --db data/y2025dec/processed/db --manifest data/y2025dec/manifest.json \
      --validation-cutoff $1 --holdout-cutoff $2 --metrics $out.json --save-probabilities > $out.log 2>&1
  done
done
