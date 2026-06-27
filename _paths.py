"""Repository and package paths for SpatioRoute experiments."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
CONFIGS_DIR = PACKAGE_ROOT / "configs"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
RESULTS_DIR = PACKAGE_ROOT / "results"

DEFAULT_PROMPT_CONFIG = CONFIGS_DIR / "prompt_config.yaml"
DEFAULT_FEW_SHOTS = PROMPTS_DIR / "few_shots.txt"
DEFAULT_SYSTEM_PROMPT = PROMPTS_DIR / "system_llm_router.txt"
