#!/usr/bin/env bash
# Bundle SpatioRoute with VLM evaluation dependencies for a standalone GitHub repo.
#
# Usage (from this directory):
#   ./scripts/prepare_github_release.sh /path/to/Qwen_playground
#
# Copies videolm/, utils/, and evaluate_*.py into ./vendor/ and adds import shims.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIOROUTE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAYGROUND="${1:-}"

if [[ -z "${PLAYGROUND}" || ! -d "${PLAYGROUND}" ]]; then
  echo "Usage: $0 /path/to/Qwen_playground"
  exit 1
fi

VENDOR="${SPATIOROUTE_ROOT}/vendor"
mkdir -p "${VENDOR}"

copy_tree() {
  local src="$1"
  local dst="$2"
  if [[ -d "${PLAYGROUND}/${src}" ]]; then
    rm -rf "${dst}"
    cp -a "${PLAYGROUND}/${src}" "${dst}"
    echo "  ✓ ${src}/"
  else
    echo "  ⚠ missing ${src}/"
  fi
}

copy_file() {
  local src="$1"
  local dst="$2"
  if [[ -f "${PLAYGROUND}/${src}" ]]; then
    cp "${PLAYGROUND}/${src}" "${dst}"
    echo "  ✓ ${src}"
  else
    echo "  ⚠ missing ${src}"
  fi
}

echo "Staging vendor code from ${PLAYGROUND} → ${VENDOR}"

copy_tree "videolm" "${VENDOR}/videolm"
copy_tree "utils" "${VENDOR}/utils"

for f in \
  evaluate_sqa.py \
  evaluate_sqa_preprompts.py \
  evaluate_sqa_simple_cot.py \
  evaluate_sqa_qwen3_text_thinking.py \
  evaluate_cot_qwen25.py \
  evaluate_cot_qwen3.py \
  evaluate_sqa_llama4.py \
  merge_vlm_preprompt_json.py \
  analyze_by_question_type.py \
  numeric_word_contains_match.py
do
  copy_file "${f}" "${VENDOR}/${f}"
done

# Root prompt config alias for legacy scripts
copy_file "prompt_config.yaml" "${SPATIOROUTE_ROOT}/prompt_config.yaml"

cat > "${SPATIOROUTE_ROOT}/sitecustomize.py" <<'PY'
"""Prepend vendor/ to sys.path when running from repo root."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
vendor = root / "vendor"
if vendor.is_dir() and str(vendor) not in sys.path:
    sys.path.insert(0, str(vendor))
    sys.path.insert(0, str(root))
PY

echo ""
echo "Done. Vendor tree: ${VENDOR}"
echo "Set PYTHONPATH when running:"
echo "  export PYTHONPATH=\"${SPATIOROUTE_ROOT}:${SPATIOROUTE_ROOT}/vendor:\$PYTHONPATH\""
echo ""
echo "Or run scripts with:"
echo "  PYTHONPATH=\"${SPATIOROUTE_ROOT}:${SPATIOROUTE_ROOT}/vendor\" python -m experiments.spatioroute.preprompts.generate_rule ..."
