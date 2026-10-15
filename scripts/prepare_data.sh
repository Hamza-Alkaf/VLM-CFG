#!/usr/bin/env bash
# Build the benchmark files that are not on the Hugging Face Hub in the form the tasks read.
#
#   bash scripts/prepare_data.sh                     # every benchmark
#   bash scripts/prepare_data.sh treebench vstar_bench_grounding
#
# Files land in $VLMCFG_DATA_DIR (default data/build/), downloads are cached in
# $VLMCFG_SOURCE_DIR (default data/build/.sources/). The VizWiz builder downloads the
# full VizWiz val image archive (3.5 GB); OCRBench v2 downloads its 10k-image test split.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; source .env; set +a; }

all=(vizwiz_grounding_val phd_sub10 treebench vstar_bench_grounding openimages refchartqa
     reasonseg_grounding mathverse_mcq ocrbench_v2_2k hallusionbench_visual amber
     detailcaps_500)
for name in "${@:-${all[@]}}"; do
    echo "== $name"
    .venv/bin/python "data/prepare_$name.py"
done
