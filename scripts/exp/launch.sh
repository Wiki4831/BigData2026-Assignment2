#!/bin/bash
# multi-seed, two-holdout runs of the 3-hop pretrained+GRU pipeline, saving probabilities
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
PY=data/model/relbench_public/venv/bin/python
for seed in 42 57 2026; do
  for pair in "2026-10-25 2026-11-01" "2026-11-01 2026-11-08"; do
    set -- $pair
    out=data/model/relbench_public/err_study/h3_val$1_s$seed
    [ -f $out.json ] && continue
    $PY scripts/relbench_asg2.py --device cuda --hops 3 --daily-loss --pretrain-epochs 3 --seed $seed \
      --validation-cutoff $1 --holdout-cutoff $2 --metrics $out.json --save-probabilities > $out.log 2>&1
  done
done
