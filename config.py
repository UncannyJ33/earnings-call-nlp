"""
config.py — Central configuration for earnings-call-nlp.

All paths, model names, and pipeline parameters live here.
No hardcoded paths elsewhere in src/ or notebooks.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

FINBERT_MODEL: str = "ProsusAI/finbert"
MAX_TOKENS: int = 512  # FinBERT max sequence length

# ---------------------------------------------------------------------------
# Return calculation
# ---------------------------------------------------------------------------

# CAR windows (trading days after earnings date)
CAR_WINDOWS: list[int] = [1, 3, 5]

# Market benchmark for abnormal return calculation
BENCHMARK_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
