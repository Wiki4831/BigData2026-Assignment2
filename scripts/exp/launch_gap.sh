#!/bin/bash
# usage: launch_gap.sh TAG VAL HOLD SEEDS... ; reuses the baseline pretrained encoder, only the GRU stage sees the gap mask
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
PY=data/model/relbench_public/venv/bin/python
tag=$1; v=$2; h=$3; shift 3
for seed in "$@"; do
  B=data/model/relbench_public/err_study/h3_val${v}_s$seed; out=data/model/relbench_public/err_study/${tag}_val${v}_s$seed
  [ -f $out.json ] && continue
  $PY scripts/relbench_asg2.py --device cuda --hops 3 --daily-loss --encoder-checkpoint $B.encoder.pt --seed $seed \
    --validation-cutoff $v --holdout-cutoff $h --metrics $out.json --save-probabilities \
    --train-gap-csv data/analysis/push_unobserved_hours.csv > $out.log 2>&1
done
