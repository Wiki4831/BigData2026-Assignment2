#!/usr/bin/env bash
# Build the RelBench environment used by scripts/predict_final.py.
# Recorded setup (verified on one machine; this script itself has not been re-run from an empty folder):
#   Python 3.11 conda env "bigdata" (see environment.yml), torch 2.11.0+cu128, venv with --system-site-packages,
#   relbench commit 5bec345fa6a424641152d8b45d91535d6a9750c4, pytorch-frame 0.3.0, torch-geometric 2.8.1,
#   pyg-lib 0.9.0+pt211cu128, sentence-transformers 6.3.0, GloVe sentence-transformers/average_word_embeddings_glove.6B.300d
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE=${ASG2_PYTHON:-python}
DIR=data/model/relbench_public
mkdir -p "$DIR"
[ -d "$DIR/upstream" ] || git clone https://github.com/stanford-star/relbench.git "$DIR/upstream"
git -C "$DIR/upstream" checkout 5bec345fa6a424641152d8b45d91535d6a9750c4
[ -d "$DIR/venv" ] || $BASE -m venv --system-site-packages "$DIR/venv"
# The venv reuses the base env's packages; install PyTorch (CUDA 12.8 build) only if it is missing.
"$DIR/venv/bin/python" -c "import torch" 2>/dev/null || \
  "$DIR/venv/bin/pip" install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$DIR/venv/bin/pip" install "$DIR/upstream" pytorch_frame==0.3.0 torch_geometric==2.8.1 sentence-transformers==6.3.0
"$DIR/venv/bin/pip" install pyg-lib==0.9.0 -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
# Cache the GloVe text embedder once; later runs are offline (HF_HUB_OFFLINE=1).
HF_HUB_CACHE="$PWD/$DIR/hf_cache" "$DIR/venv/bin/python" - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/average_word_embeddings_glove.6B.300d")
PY
echo "RelBench environment ready: $DIR/venv/bin/python"
