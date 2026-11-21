#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ASG2_PYTHON:-python}"
cd "$PROJECT_DIR"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python unavailable: $PYTHON. Set ASG2_PYTHON to your environment's Python." >&2
    exit 1
fi
case "${1:-help}" in
    env) "$PYTHON" scripts/check_setup.py ;;
    download) shift; "$PYTHON" -u scripts/download_gharchive.py "$@" ;;
    preprocess)
        shift
        "$PYTHON" -u scripts/preprocess.py --raw data/raw/gharchive --out data/processed "$@"
        ;;
    check) shift; "$PYTHON" scripts/check_setup.py --tables data/processed/db "$@" ;;
    evaluate) shift; "$PYTHON" scripts/evaluate.py "$@" ;;
    train|predict) action="$1"; shift; "$PYTHON" -u scripts/train_model.py "$action" "$@" ;;
    *)
        echo "Usage: bash run.sh {env|download|preprocess|check|evaluate|train|predict} [arguments]"
        echo "Download defaults: 2026-10-01 through the latest completed UTC day."
        ;;
esac
