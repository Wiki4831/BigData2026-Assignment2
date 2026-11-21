#!/bin/bash
# full pipeline (3-hop pretrain + GRU), a new test week, 3 seeds. usage: launch_week.sh VAL HOLD
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
PY=data/model/relbench_public/venv/bin/python
for seed in 42 57 2026; do
  out=data/model/relbench_public/err_study/h3_val$1_s$seed
  [ -f $out.json ] && continue
  $PY scripts/relbench_asg2.py --device cuda --hops 3 --daily-loss --pretrain-epochs 3 --seed $seed \
    --validation-cutoff $1 --holdout-cutoff $2 --metrics $out.json --save-probabilities > $out.log 2>&1
done
