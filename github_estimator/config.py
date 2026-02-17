"""Configuration constants for GitHub User Estimation project."""

API_BASE_URL = "https://api.github.com/user"

# Rate limiting: all tokens share ONE account's 5,000 req/hr limit
RATE_LIMIT_PER_HOUR = 5_000
RATE_LIMIT_BUFFER = 100  # Keep this many requests in reserve
CONCURRENT_REQUESTS = 15  # Conservative to avoid bursts triggering abuse detection

BOOTSTRAP_ITERATIONS = 1000
DEFAULT_REPETITIONS = 50

# Sampling rates for validation experiments
SAMPLING_RATES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

# --- Pool-based estimation strategy ---
# ~5 hr budget: 8K ground truth + 16K pool = 24K calls
FULL_SPACE_POOL_SIZE = 16_000

# Budget levels for partitioned analysis
PARTITION_BUDGETS = [1_000, 2_000, 4_000, 8_000, 16_000]
# Yields:              16       8       4       2       1  independent runs

# Current frontier (approximate max GitHub user ID as of Feb 15, 2026)
FRONTIER_M = 261_712_000

# Database paths
TOKENS_DB_PATH = "tokens.db"
DATA_DB_PATH = "github_data.db"

# Output paths
OUTPUT_DIR = "output"
FIGURES_DIR = "output/figures"
DATA_OUTPUT_DIR = "output/data"
TABLES_DIR = "output/report/tables"

# --- Validation Strata (exhaustively collected as ground truth) ---
# 8 strata × 1,000 IDs = 8,000 total (~1.6 hrs at 5K/hr)
VALIDATION_STRATA = {
    "V1": {"start": 1, "end": 1_000, "size": 1_000, "description": "Oldest accounts"},
    "V2": {"start": 50_001, "end": 51_000, "size": 1_000, "description": "Early adopter era"},
    "V3": {"start": 1_000_001, "end": 1_001_000, "size": 1_000, "description": "Early growth period"},
    "V4": {"start": 10_000_001, "end": 10_001_000, "size": 1_000, "description": "2012 era accounts"},
    "V5": {"start": 50_000_001, "end": 50_001_000, "size": 1_000, "description": "2016-2017 era"},
    "V6": {"start": 100_000_001, "end": 100_001_000, "size": 1_000, "description": "2019-2020 era"},
    "V7": {"start": 200_000_001, "end": 200_001_000, "size": 1_000, "description": "2023-2024 era"},
    "V8": {"start": 260_000_001, "end": 260_001_000, "size": 1_000, "description": "2026 accounts"},
}

# --- Full Space Strata (for final estimation) ---
FULL_SPACE_STRATA = {
    "F1": {"start": 1, "end": 10_000_000, "size": 10_000_000},
    "F2": {"start": 10_000_001, "end": 50_000_000, "size": 40_000_000},
    "F3": {"start": 50_000_001, "end": 100_000_000, "size": 50_000_000},
    "F4": {"start": 100_000_001, "end": 150_000_000, "size": 50_000_000},
    "F5": {"start": 150_000_001, "end": 200_000_000, "size": 50_000_000},
    "F6": {"start": 200_000_001, "end": 250_000_000, "size": 50_000_000},
    "F7": {"start": 250_000_001, "end": FRONTIER_M, "size": FRONTIER_M - 250_000_000},
}
