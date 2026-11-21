#!/bin/bash
# usage: launch_group.sh TAG GROUP   (full pretrain per run; the graph differs from the baseline)
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
PY=data/model/relbench_public/venv/bin/python
tag=$1; G=$2
for seed in 42 57 2026; do
  for pair in "2026-10-25 2026-11-01" "2026-11-01 2026-11-08"; do
    set -- $pair
    out=data/model/relbench_public/err_study/${tag}_val$1_s$seed
    [ -f $out.json ] && continue
    $PY scripts/relbench_asg2.py --device cuda --hops 3 --daily-loss --pretrain-epochs 3 --seed $seed --repo-group $G \
      --validation-cutoff $1 --holdout-cutoff $2 --metrics $out.json --save-probabilities > $out.log 2>&1
  done
done
