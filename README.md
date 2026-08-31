# CFB Score Predictor

## Setup
```
pip install cfbd pandas
export CFBD_API_KEY="your_key_here"
```

## Run
```
cd features
python build_feature_table.py
```
Writes `data/training_table.csv` — one row per game, 2022-2025, with
pre-game features for both teams (SP+, talent composite, power rating,
recency-weighted sample_weight column) and targets (margin, total_points,
home_win). No leakage — power ratings are snapshotted BEFORE each game.

## Files
- `power_rating.py` — Python translation of your Excel power-rating workbook.
  Weights are FIXED (verified bit-for-bit against the workbook's own Alabama
  row: adjusted=-57.2, sos_quality_pts=13.5 for the Florida St game — matches).
- `cfbd_pull.py` — CFBD API client wrapper + dynamic tier assignment
  (SP+ rank + AP rank cutoffs, since your manual tiers can't be replicated
  programmatically across 4 seasons).
- `build_feature_table.py` — merges everything into the training table.

## One thing to verify on first real run
In `build_feature_table.py::ap_ranks_by_week()`, the AP poll match is a
loose `"ap" in poll.poll.lower()` check — I couldn't confirm CFBD's exact
poll name string without a live key. Print `poll.poll` once and tighten
to an exact match if you want it more precise.

## Not built yet (next steps)
- XGBoost training script (spread + score regression)
- Time-based train/val/test split (train 2022-2023, val 2024, test 2025)
- Weekly fixed-vs-learned power-rating weight shadow comparison
- Roster-continuity/portal-turnover feature (deferred per your call)
