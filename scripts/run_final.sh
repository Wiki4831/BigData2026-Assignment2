#!/usr/bin/env bash
# Final pipeline: download -> official preprocess -> train 3 seeds -> output_{studentID}.csv -> format check.
# usage: bash scripts/run_final.sh STUDENT_ID [END_UTC_DATE]
#   END_UTC_DATE default 2026-11-20: the last completed UTC day before the test cutoff (2026-11-21 16:00 UTC).
# Environment overrides (defaults give the standard layout, so a fresh checkout needs none of them):
#   RAW_DIR       GH Archive hour files          (default data/raw/gharchive)
#   PROC_DIR      official preprocess output     (default data/processed; tables go to $PROC_DIR/db)
#   MANIFEST      download manifest              (default data/download_manifest.json)
#   SKIP_DOWNLOAD=1   reuse the files already in RAW_DIR and MANIFEST
#   ASG2_PYTHON   Python for preprocessing and evaluation (default: `python` on PATH, i.e. the activated conda env)
#   GNN_PYTHON    Python of the RelBench venv    (default data/model/relbench_public/venv/bin/python)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ID=${1:?student id}; END=${2:-2026-11-20}
RAW=${RAW_DIR:-data/raw/gharchive}; PROC=${PROC_DIR:-data/processed}; MANIFEST=${MANIFEST:-data/download_manifest.json}
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
BASE=${ASG2_PYTHON:-python}
GNN=${GNN_PYTHON:-data/model/relbench_public/venv/bin/python}
[ -x "$GNN" ] || { echo "RelBench venv missing: run bash scripts/setup_relbench.sh first" >&2; exit 1; }
if [ "${SKIP_DOWNLOAD:-0}" != 1 ]; then
  $BASE -u scripts/download_gharchive.py --start 2026-10-01 --end "$END" --raw "$RAW" --manifest "$MANIFEST"
fi
$BASE -u scripts/preprocess.py --raw "$RAW" --out "$PROC"
$GNN -u scripts/predict_final.py --student-id "$ID" --output "output_$ID.csv" --db "$PROC/db" --manifest "$MANIFEST" \
  --seeds 42 --deterministic
$BASE scripts/evaluate.py --pred "output_$ID.csv" --input submission/test_input.csv
