from __future__ import annotations
from typing import Any
from pathlib import Path
from functools import lru_cache

import joblib  # pip install joblib

MODEL_PATH = Path("cost_model.pkl")

@lru_cache(maxsize=1)
def _get_model():
    # Only load pickle artifacts you trust (see security note below).
    return joblib.load(MODEL_PATH)
