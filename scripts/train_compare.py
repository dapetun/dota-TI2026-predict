"""CLI: compare XGBoost, CatBoost and blend on current OpenDota data."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.train_compare import run_model_compare


if __name__ == "__main__":
    run_model_compare()
