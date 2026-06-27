#!/usr/bin/env python3
"""Per-question-type accuracy table (SQA3D categories)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_by_question_type import main as analyze_main  # noqa: E402


if __name__ == "__main__":
    analyze_main()
