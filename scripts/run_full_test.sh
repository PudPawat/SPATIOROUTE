#!/usr/bin/env bash
# Full SQA test split — SpatioRoute-R, SpatioRoute-L, baseline (no sample cap).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SPLIT="${SPLIT:-test}"
MODEL="${VLM_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
ROUTER="${ROUTER_LLM:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_DIR="experiments/spatioroute/results"
mkdir -p "$OUT_DIR"

MODEL_TAG="$(echo "$MODEL" | tr '/:' '_' | tr '[:upper:]' '[:lower:]')"

echo "=== [1/5] SpatioRoute-R preprompts ==="
python -m experiments.spatioroute.preprompts.generate_rule \
  --split "$SPLIT" \
  --output "${OUT_DIR}/preprompts_r_${SPLIT}.json"

echo "=== [2/5] SpatioRoute-L preprompts ==="
python -m experiments.spatioroute.preprompts.generate_llm \
  --split "$SPLIT" \
  --model-name "$ROUTER" \
  --load-in-4bit \
  --checkpoint-every 25 \
  --output "${OUT_DIR}/preprompts_l_${SPLIT}.json"

echo "=== [3/5] Baseline VLM ==="
python -m experiments.spatioroute.eval.baseline \
  --split "$SPLIT" \
  --model-name "$MODEL" \
  --load-in-4bit \
  --save-interval 50 \
  --output "${OUT_DIR}/baseline_${MODEL_TAG}_${SPLIT}.json"

echo "=== [4/5] SpatioRoute-R VLM ==="
python -m experiments.spatioroute.eval.routed \
  --preprompt-json "${OUT_DIR}/preprompts_r_${SPLIT}.json" \
  --split "$SPLIT" \
  --model-name "$MODEL" \
  --load-in-4bit \
  --save-interval 50 \
  --output "${OUT_DIR}/spatioroute_r_${MODEL_TAG}_${SPLIT}.json"

echo "=== [5/5] SpatioRoute-L VLM ==="
python -m experiments.spatioroute.eval.routed \
  --preprompt-json "${OUT_DIR}/preprompts_l_${SPLIT}.json" \
  --split "$SPLIT" \
  --model-name "$MODEL" \
  --load-in-4bit \
  --save-interval 50 \
  --output "${OUT_DIR}/spatioroute_l_${MODEL_TAG}_${SPLIT}.json"

echo "=== Analysis ==="
python -m experiments.spatioroute.analysis.by_question_type \
  "${OUT_DIR}/baseline_${MODEL_TAG}_${SPLIT}.json" \
  "${OUT_DIR}/spatioroute_r_${MODEL_TAG}_${SPLIT}.json" \
  "${OUT_DIR}/spatioroute_l_${MODEL_TAG}_${SPLIT}.json" \
  --save

echo "Done."
