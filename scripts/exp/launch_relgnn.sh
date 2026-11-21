#!/bin/bash
# usage: launch_relgnn.sh TAG HOPS VAL HOLD SEEDS...
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
PY=data/model/relbench_public/venv/bin/python
tag=$1; hops=$2; v=$3; h=$4; shift 4
for seed in "$@"; do
  out=data/model/relbench_public/err_study/${tag}_val${v}_s$seed
  [ -f $out.json ] && continue
  $PY scripts/relbench_asg2.py --device cuda --hops $hops --daily-loss --pretrain-epochs 3 --seed $seed --backbone relgnn \
    --validation-cutoff $v --holdout-cutoff $h --metrics $out.json --save-probabilities > $out.log 2>&1
done
