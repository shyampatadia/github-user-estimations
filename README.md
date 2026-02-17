# GitHub User Estimation

Estimates the total number of valid GitHub users using stratified random sampling with bootstrap inference.

GitHub assigns sequential integer IDs to users, but not all IDs are active — some are deleted or suspended. This project samples ~100K IDs across the full ID space (~262M), measures the validity rate per stratum, and extrapolates to estimate the true total.

## Setup

```bash
uv pip install -r requirements.txt
```

Add your GitHub PATs to `tokens.db` (SQLite):

```sql
CREATE TABLE tokens (
    id INTEGER PRIMARY KEY,
    token TEXT NOT NULL,
    description TEXT,
    scopes TEXT,
    created_at TIMESTAMP
);

INSERT INTO tokens (id, token, description) VALUES (1, 'ghp_xxxx...', 'PAT 1');
-- All tokens share one account's 5,000 req/hr limit
```

## Usage

```bash
uv run python run.py                      # Full pipeline (~20 hrs collection + analysis)
uv run python run.py --phase 1            # Ground truth only
uv run python run.py --phase 3            # Full-space sampling only
uv run python run.py --skip-phase1 --skip-phase3   # Analysis + figures only (no API calls)
```

Collection is resumable — restart anytime and it picks up where it left off.

## Project Structure

```
github_estimator/
  config.py           Strata definitions and constants
  database.py         SQLite schema and operations
  token_manager.py    PAT rotation and rate limiting
  crawler.py          Async GitHub API fetching
  sampler.py          Stratified random sampling
  estimator.py        Statistical estimation and bootstrap CI
  validation.py       Validation experiments on ground truth
  visualization.py    Figure generation
  main.py             Pipeline orchestration
  utils.py            Helpers
```
