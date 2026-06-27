# SpatioRoute: Dynamic Prompt Routing for Zero-Shot Spatial Reasoning

[![Paper](https://img.shields.io/badge/arXiv-2605.18209-b31b1b)](https://arxiv.org/abs/2605.18209)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://pudpawat.github.io/SPATIOROUTE-Dynamic-Prompt-Routing-for-Zero-Shot-Spatial-Reasoning/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Project page:** https://pudpawat.github.io/SPATIOROUTE-Dynamic-Prompt-Routing-for-Zero-Shot-Spatial-Reasoning/

Reproducible experiment code for the CVPR 2026 workshop paper  
**[SPATIOROUTE: Dynamic Prompt Routing for Zero-Shot Spatial Reasoning](https://arxiv.org/abs/2605.18209)**  
(Pawat Chunhachatrachai, Gueter Josmy Faure, Hung-Ting Su, Winston H. Hsu).

> **Publishing this repo on GitHub?** See [GITHUB_SETUP.md](GITHUB_SETUP.md) for `git push` steps and enabling GitHub Pages from the `docs/` folder.

This folder organizes the original repo scripts into a clear pipeline:

| Paper method | Code entry point | Routing at inference |
|---|---|---|
| **Fixed baseline** | `eval/baseline.py` | One YAML prompt for all questions |
| **SpatioRoute-R** | `preprompts/generate_rule.py` → `eval/routed.py` | Rule-based: SQA type → YAML template |
| **SpatioRoute-L** | `preprompts/generate_llm.py` → `eval/routed.py` | Same rules + small text LLM + few-shots |
| **CoT (Think it Twice)** | `eval/cot.py` | Uniform two-pass CoT (comparison) |

All VLM evaluation uses **video frames only** (no 3D point clouds). SpatioRoute-L routes from **question + situation text** without watching video at routing time.

---

## Directory layout

```
experiments/spatioroute/
├── README.md                 ← this file
├── configs/
│   ├── prompt_config.yaml    ← VLM prompt templates (What→details_scene, …)
│   └── experiments.yaml      ← default paths and routing table
├── prompts/
│   ├── few_shots.txt         ← few-shot demos for SpatioRoute-L (paper Appendix style)
│   └── system_llm_router.txt ← system prompt for the router LLM
├── routing/                  ← shared classification + template logic
├── preprompts/
│   ├── generate_rule.py      ← SpatioRoute-R
│   ├── generate_llm.py       ← SpatioRoute-L (new, self-contained)
│   └── merge.py              ← merge preprompt JSON shards
├── eval/
│   ├── baseline.py           ← fixed-prompt VLM eval
│   ├── routed.py             ← VLM eval with routed preprompts
│   └── cot.py                ← two-pass CoT comparison
├── analysis/
│   └── by_question_type.py   ← SQA3D category breakdown
├── scripts/
│   └── run_smoke_test.sh     ← 50-sample end-to-end test
└── results/                  ← put JSON outputs here
```

Legacy scripts at the repo root (`generate_vlm_preprompts.py`, `evaluate_sqa_preprompts.py`, …) still work; this package wraps them with stable paths and paper naming.

---

## 1. Environment setup

Run all commands from the **repository root** (`Qwen_playground/`).

### 1.1 System requirements

- Linux with NVIDIA GPU (CUDA)
- Python 3.10+ recommended
- ~8 GB+ VRAM for Qwen2-VL-2B with 4-bit quantization (more for 7B or full precision)
- SQA3D dataset under `dataset/SQA/` (questions, annotations, videos)

### 1.2 Create virtual environment

```bash
cd /path/to/Qwen_playground

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Optional (quantization):

```bash
pip install bitsandbytes>=0.41.0
```

Optional (faster video decoding):

```bash
pip install decord
```

### 1.3 Dataset

Place the ScanNet SQA3D release so these paths exist:

```
dataset/SQA/
├── sqa_task/balanced/v1_balanced_questions_test_scannetv2.json
├── sqa_task/balanced/v1_balanced_sqa_annotations_test_scannetv2.json
└── video/<scene_id>.mp4
```

Same layout for `train` / `val` splits if needed.

### 1.4 Hugging Face models

Models download automatically on first run. For gated models (Llama), log in:

```bash
huggingface-cli login
```

---

## 2. Routing table (SpatioRoute-R and SpatioRoute-L)

Question type is inferred from the **first word(s)** of each SQA question (SQA3D convention):

| SQA category | Example prefix | Template (`prompt_config.yaml`) |
|---|---|---|
| What | "What …" | `details_scene` |
| How many | "How many …" | `details_scene` |
| Which | "Which …" | `details_scene` |
| Is | "Is …" | `step_by_step` |
| Can | "Can …" | `scene_understanding` |
| Others | Where, How, … | `focus_instructions` |

**SpatioRoute-R** fills the YAML template directly (deterministic).  
**SpatioRoute-L** uses the same route, then a **text-only LLM** rewrites the prompt using `prompts/few_shots.txt` and `prompts/system_llm_router.txt`.

Default router LLM: `Qwen/Qwen2.5-0.5B-Instruct` (fast, no vision).

---

## 3. Running experiments

### 3.1 Quick smoke test (~50 samples)

```bash
source venv/bin/activate
chmod +x experiments/spatioroute/scripts/run_smoke_test.sh
./experiments/spatioroute/scripts/run_smoke_test.sh
```

Override model or sample count:

```bash
VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct MAX_SAMPLES=20 ./experiments/spatioroute/scripts/run_smoke_test.sh
```

### 3.2 Full test split — step by step

#### A. Fixed baseline

One prompt (`scene_understanding`) for every question:

```bash
python -m experiments.spatioroute.eval.baseline \
  --split test \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --prompt-name scene_understanding \
  --load-in-4bit \
  --max-frames 8 \
  --output experiments/spatioroute/results/baseline_qwen2vl2b_test.json
```

#### B. SpatioRoute-R (rule-based)

**Step B1 — generate preprompts** (no GPU required):

```bash
python -m experiments.spatioroute.preprompts.generate_rule \
  --split test \
  --output experiments/spatioroute/results/preprompts_r_test.json
```

**Step B2 — VLM evaluation**:

```bash
python -m experiments.spatioroute.eval.routed \
  --preprompt-json experiments/spatioroute/results/preprompts_r_test.json \
  --split test \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --load-in-4bit \
  --max-frames 8 \
  --save-interval 50 \
  --output experiments/spatioroute/results/spatioroute_r_qwen2vl2b_test.json
```

#### C. SpatioRoute-L (LLM + few-shots)

**Step C1 — generate preprompts** (text LLM; resume-safe):

```bash
python -m experiments.spatioroute.preprompts.generate_llm \
  --split test \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --load-in-4bit \
  --output experiments/spatioroute/results/preprompts_l_test.json \
  --checkpoint-every 25
```

Use a larger router if desired:

```bash
python -m experiments.spatioroute.preprompts.generate_llm \
  --model-preset qwen25-1.5b-instruct \
  --split test \
  --output experiments/spatioroute/results/preprompts_l_test.json
```

Custom few-shots / system prompt:

```bash
python -m experiments.spatioroute.preprompts.generate_llm \
  --few-shots-file experiments/spatioroute/prompts/few_shots.txt \
  --system-prompt-file experiments/spatioroute/prompts/system_llm_router.txt \
  --output experiments/spatioroute/results/preprompts_l_test.json
```

**Step C2 — VLM evaluation** (same as SpatioRoute-R):

```bash
python -m experiments.spatioroute.eval.routed \
  --preprompt-json experiments/spatioroute/results/preprompts_l_test.json \
  --split test \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --load-in-4bit \
  --output experiments/spatioroute/results/spatioroute_l_qwen2vl2b_test.json
```

#### D. CoT comparison (Think it Twice)

Simple two-pass CoT on Qwen2-VL (paper finding: often hurts vs routing):

```bash
python -m experiments.spatioroute.eval.cot \
  --backend qwen2_2b \
  --split test \
  --load-in-4bit \
  --output experiments/spatioroute/results/cot_qwen2vl2b_test.json
```

Other backends: `qwen2_7b`, `qwen25`, `qwen3`, `llama`.

### 3.3 Other VLM families

Use the same preprompt JSON; only change `--model-name`:

```bash
# Qwen2.5-VL
python -m experiments.spatioroute.eval.routed \
  --preprompt-json experiments/spatioroute/results/preprompts_l_test.json \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --load-in-8bit \
  --output experiments/spatioroute/results/spatioroute_l_qwen25vl3b_test.json

# Qwen3-VL
python -m experiments.spatioroute.eval.routed \
  --preprompt-json experiments/spatioroute/results/preprompts_l_test.json \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --load-in-4bit \
  --output experiments/spatioroute/results/spatioroute_l_qwen3vl2b_test.json

# Llama 3.2 Vision
python -m experiments.spatioroute.eval.routed \
  --preprompt-json experiments/spatioroute/results/preprompts_l_test.json \
  --model-name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --no-quantization \
  --output experiments/spatioroute/results/spatioroute_l_llama32_11b_test.json
```

If 8-bit quantization fails with vision `CB` errors, use `--no-quantization`.

### 3.4 Resume and merge

**VLM eval** resumes automatically when `--output` already exists (via `evaluate_sqa` checkpointing).

**SpatioRoute-L preprompt generation** skips completed `question_id`s when reusing `--output`.

**Merge preprompt shards**:

```bash
python -m experiments.spatioroute.preprompts.merge \
  --output experiments/spatioroute/results/preprompts_l_test_merged.json \
  part_a.json part_b.json part_c.json
```

---

## 4. Analysis

### Per-question-type accuracy (SQA3D categories)

```bash
python -m experiments.spatioroute.analysis.by_question_type \
  experiments/spatioroute/results/baseline_qwen2vl2b_test.json \
  experiments/spatioroute/results/spatioroute_r_qwen2vl2b_test.json \
  experiments/spatioroute/results/spatioroute_l_qwen2vl2b_test.json \
  --save
```

Uses the **contains** metric (`correct_contains`) by default.

### Research tables (repo root)

```bash
python make_research_table.py --config research_data_files.json --metric contains_numword
```

Point `research_data_files.json` at your result paths under `experiments/spatioroute/results/`.

---

## 5. Customizing prompts

| File | Purpose |
|---|---|
| `configs/prompt_config.yaml` | Template bodies for each routed style |
| `prompts/few_shots.txt` | In-context examples for SpatioRoute-L |
| `prompts/system_llm_router.txt` | Router LLM system instructions |

After editing templates, regenerate preprompts (R and/or L) before re-running VLM eval.

---

## 6. Mapping to paper claims

- **Zero-shot / no fine-tuning**: all scripts use off-the-shelf VLMs and a frozen router LLM.
- **No 3D sensor input**: only uniformly sampled video frames (`--max-frames 8` default).
- **Up to ~5% gain vs fixed prompt**: compare `baseline_*.json` vs `spatioroute_r_*.json` or `spatioroute_l_*.json` on the same model.
- **CoT degradation on Qwen**: compare `spatioroute_*` vs `cot_*` on Qwen2 / Qwen2.5 / Qwen3 backends.

---

## 7. Troubleshooting

| Issue | Fix |
|---|---|
| CUDA OOM | `--load-in-4bit`, reduce `--max-frames`, smaller VLM |
| `CB` attribute error (quantization) | `--no-quantization` |
| Missing videos | Check `dataset/SQA/video/<scene_id>.mp4` |
| Empty preprompt rows | Re-run L generator; skip rows starting with `ERROR:` |
| Slow L generation | Default `Qwen2.5-0.5B`; lower `--max-new-tokens` |

---

## Citation

```bibtex
@article{chunhachatrachai2026spatioroute,
  title={SPATIOROUTE: Dynamic Prompt Routing for Zero-Shot Spatial Reasoning},
  author={Chunhachatrachai, Pawat and Faure, Gueter Josmy and Su, Hung-Ting and Hsu, Winston H.},
  journal={arXiv preprint arXiv:2605.18209},
  year={2026}
}
```
