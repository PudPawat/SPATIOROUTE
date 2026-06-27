#!/usr/bin/env bash
# End-to-end SpatioRoute smoke test (50 samples). Run from repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SPLIT="${SPLIT:-test}"
MAX="${MAX_SAMPLES:-50}"
MODEL="${VLM_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
OUT_DIR="experiments/spatioroute/results"
mkdir -p "$OUT_DIR"

echo "=== SpatioRoute-R: generate preprompts (${MAX} samples) ==="
python -m experiments.spatioroute.preprompts.generate_rule \
  --split "$SPLIT" \
  --max-samples "$MAX" \
  --output "${OUT_DIR}/preprompts_r_${SPLIT}_smoke.json"

echo "=== SpatioRoute-L: generate preprompts (${MAX} samples) ==="
python -m experiments.spatioroute.preprompts.generate_llm \
  --split "$SPLIT" \
  --max-samples "$MAX" \
  --load-in-4bit \
  --output "${OUT_DIR}/preprompts_l_${SPLIT}_smoke.json"

echo "=== Baseline VLM eval ==="
python -m experiments.spatioroute.eval.baseline \
  --split "$SPLIT" \
  --model-name "$MODEL" \
  --max-samples "$MAX" \
  --load-in-4bit \
  --output "${OUT_DIR}/baseline_smoke.json"

echo "=== SpatioRoute-R VLM eval ==="
python -m experiments.spatioroute.eval.routed \
  --preprompt-json "${OUT_DIR}/preprompts_r_${SPLIT}_smoke.json" \
  --split "$SPLIT" \
  --model-name "$MODEL" \
  --max-samples "$MAX" \
  --load-in-4bit \
  --output "${OUT_DIR}/spatioroute_r_smoke.json"

echo "=== SpatioRoute-L VLM eval ==="
python -m experiments.spatioroute.eval.routed \
  --preprompt-json "${OUT_DIR}/preprompts_l_${SPLIT}_smoke.json" \
  --split "$SPLIT" \
  --model-name "$MODEL" \
  --max-samples "$MAX" \
  --load-in-4bit \
  --output "${OUT_DIR}/spatioroute_l_smoke.json"

echo "=== Per-type analysis ==="
python -m experiments.spatioroute.analysis.by_question_type \
  "${OUT_DIR}/baseline_smoke.json" \
  "${OUT_DIR}/spatioroute_r_smoke.json" \
  "${OUT_DIR}/spatioroute_l_smoke.json"

echo "Done. Results in ${OUT_DIR}/"
