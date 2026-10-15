#!/usr/bin/env bash
# Build the project environment in ./.venv from the committed uv.lock, fetch the NLTK
# corpus the OCRBench v2 scorer needs, and create .env from .env.example.
#
#   bash scripts/setup_env.sh                  # core environment
#   WITH_LONGFORM=1 bash scripts/setup_env.sh  # + AMBER / DetailCaps scorers
#
# Requires uv (https://docs.astral.sh/uv/getting-started/installation/) and an
# NVIDIA driver recent enough for CUDA 11.8 wheels.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || {
    echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

extras=()
if [ "${WITH_LONGFORM:-0}" = 1 ]; then
    extras+=(--extra longform)
fi
# --frozen: install exactly what uv.lock pins, never re-resolve.
uv sync --frozen --python 3.11 "${extras[@]}"

# nltk only downloads into a directory on its data path, and exits 0 even when it refuses.
nltk_dir="$PWD/.cache/nltk_data"
mkdir -p "$nltk_dir"
NLTK_DATA="$nltk_dir" .venv/bin/python -m nltk.downloader -q -d "$nltk_dir" wordnet
[ -e "$nltk_dir/corpora/wordnet.zip" ] || [ -d "$nltk_dir/corpora/wordnet" ] || {
    echo "NLTK wordnet download failed (needed by the OCRBench v2 scorer)" >&2
    exit 1
}
if [ "${WITH_LONGFORM:-0}" = 1 ]; then
    # the AMBER scorer tokenizes and POS-tags responses
    NLTK_DATA="$nltk_dir" .venv/bin/python -m nltk.downloader -q -d "$nltk_dir" \
        punkt_tab averaged_perceptron_tagger_eng
    [ -e "$nltk_dir/tokenizers/punkt_tab.zip" ] \
        && [ -e "$nltk_dir/taggers/averaged_perceptron_tagger_eng.zip" ] || {
        echo "NLTK punkt_tab / tagger download failed (needed by the AMBER scorer)" >&2
        exit 1
    }
fi

# .env holds tokens: readable by the owner only.
[ -f .env ] || install -m 600 .env.example .env

.venv/bin/python - <<'PY'
import torch

print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"GPU: {name} (sm_{major}{minor})")
    if major < 8:
        print("No bf16 tensor cores on this GPU: Qwen3-VL and Molmo2 presets run in fp16.")
PY
echo "Environment ready: source .venv/bin/activate"
