#!/bin/bash
# usage: launch_rank.sh TAG "extra args"   (reuses the baseline pretrained encoder of each val/seed)
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
PY=data/model/relbench_public/venv/bin/python
tag=$1; EXTRA="$2"
for seed in 42 57 2026; do
  for pair in "2026-10-25 2026-11-01" "2026-11-01 2026-11-08"; do
    set -- $pair; v=$1; h=$2
    B=data/model/relbench_public/err_study/h3_val${v}_s$seed
    out=data/model/relbench_public/err_study/${tag}_val${v}_s$seed
    [ -f $out.json ] && continue
    $PY scripts/relbench_asg2.py --device cuda --hops 3 --daily-loss --encoder-checkpoint $B.encoder.pt --seed $seed \
      --validation-cutoff $v --holdout-cutoff $h --metrics $out.json --save-probabilities $EXTRA > $out.log 2>&1
  done
done
