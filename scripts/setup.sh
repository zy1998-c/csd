#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REV=b8a03b3ce34fb4a96001df545a56558d75a078a3
BASE="${ROOT}/vendor/RWKU"
if [ ! -d "${BASE}/.git" ]; then
  mkdir -p "${ROOT}/vendor"
  git clone --depth 1 https://github.com/jinzhuoran/RWKU.git "${BASE}"
fi
if [ "$(git -C "${BASE}" rev-parse HEAD)" != "$REV" ]; then
  git -C "${BASE}" fetch --depth 1 origin "$REV"
  git -C "${BASE}" checkout --detach "$REV"
fi
python -m pip install -r "${ROOT}/requirements.txt"
bash "${ROOT}/scripts/apply_overlay.sh" "${BASE}/LLaMA-Factory"
export NLTK_DATA="${ROOT}/vendor/nltk_data"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
python -c 'import nltk, os; assert nltk.download("punkt", download_dir=os.environ["NLTK_DATA"]), "Could not download NLTK punkt"'
PYTHONPATH="${BASE}/LLaMA-Factory/src" python -c 'from llmtuner.train.csd import run_csd; from llmtuner.eval import eval_forget; print("CSD/RWKU imports OK")'
