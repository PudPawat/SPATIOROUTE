"""Prepend vendor/ to sys.path when running from repo root."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
vendor = root / "vendor"
if vendor.is_dir() and str(vendor) not in sys.path:
    sys.path.insert(0, str(vendor))
    sys.path.insert(0, str(root))
