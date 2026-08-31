"""
Grades a saved predictions CSV against actual final scores, once those
games have been played. Pulls real results from CFBD and joins them onto
your predictions by game_id — no re-computation of features needed, this
just checks how the existing predictions held up.

Usage:
    cd features
    python ../models/grade_predictions.py ../data/predictions_2026_week1.csv

    # Or override season/week explicitly instead of parsing from filename:
    python ../models/grade_predictions.py ../data/predictions_2026_week1.csv --season 2026 --week 1
"""

import sys
import re
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from cfbd_pull import CFBDClient


def parse_season_week_from_filename(path: str):
    """predictions_2026_week1.csv -> (2026, 1)"""
    match = re.search(r"predictions_(\d{4})_week(\d+)", path)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()

    season = args.season
    week = args.week
    if season is None or week is None:
        parsed_season, parsed_week = parse_season_week_from_filename(args.predictions_csv)
        season = season or parsed_season
        week = week or parsed_week
    if season is None:
        raise SystemExit("Couldn't determine season — pass --season explicitly.")

    preds = pd.read_csv(args.predictions_csv)
    print(f"Loaded {len(preds)} predictions from {args.predictions_csv}")

    client = CFBDClient()
    print(f"Pulling actual {season} results to check against...")
    all_games = client.get_games(season)

    actuals = pd.DataFrame([{
        "game_id": g.id,
        "actual_home_points": g.home_points,
        "actual_away_points": g.away_points,
    } for g in all_games])

    merged = preds.merge(actuals, on="game_id", how="left")

    played = merged[merged["actual_home_points"].notna()].copy()
    not_played = merged[merged["actual_home_points"].isna()]

    if not_played.shape[0] > 0:
        print(f"{len(not_played)} of {len(merged)} games haven't finished yet — grading only the {len(played)} that have.\n")

    if played.empty:
        print("No games in this file have final scores yet. Nothing to grade.")
        return

    played["actual_margin"] = played["actual_home_points"] - played["actual_away_points"]
    played["actual_total"] = played["actual_home_points"] + played["actual_away_points"]
    played["actual_home_win"] = (played["actual_margin"] > 0).astype(int)

    played["margin_error"] = played["predicted_margin"] - played["actual_margin"]
    played["abs_margin_error"] = played["margin_error"].abs()
    played["home_score_error"] = played["predicted_home_score"] - played["actual_home_points"]
    played["away_score_error"] = played["predicted_away_score"] - played["actual_away_points"]

    predicted_home_win = (played["home_win_prob"] >= 0.5).astype(int)
    played["win_correct"] = (predicted_home_win == played["actual_home_win"]).astype(int)

    # ---- Overall model performance this week ----
    margin_rmse = np.sqrt((played["margin_error"] ** 2).mean())
    win_acc = played["win_correct"].mean()
    print(f"=== Model performance on {len(played)} graded games ===")
    print(f"Margin RMSE: {margin_rmse:.2f} points")
    print(f"Mean absolute margin error: {played['abs_margin_error'].mean():.2f} points")
    print(f"Win/loss accuracy: {win_acc:.1%}\n")

    # ---- Compare to Vegas on the same games, if lines were present ----
    if "vegas_implied_margin" in played.columns:
        has_line = played["vegas_implied_margin"].notna()
        if has_line.sum() > 0:
            vegas_error = played.loc[has_line, "vegas_implied_margin"] - played.loc[has_line, "actual_margin"]
            vegas_rmse = np.sqrt((vegas_error ** 2).mean())
            print(f"=== Vegas comparison on {has_line.sum()} games with lines ===")
            print(f"Model margin RMSE: {margin_rmse:.2f}")
            print(f"Vegas margin RMSE: {vegas_rmse:.2f}")
            diff = vegas_rmse - margin_rmse
            print(f"Model {'beat' if diff > 0 else 'lost to'} Vegas by {abs(diff):.2f} points RMSE this week.\n")

    # ---- Best and worst predictions ----
    print("=== Best predictions (smallest margin error) ===")
    cols = ["home_team", "away_team", "predicted_margin", "actual_margin", "margin_error"]
    print(played.nsmallest(5, "abs_margin_error")[cols].to_string(index=False))
    print()

    print("=== Worst predictions (biggest margin error) ===")
    print(played.nlargest(5, "abs_margin_error")[cols].to_string(index=False))
    print()

    # ---- Save graded output for dashboard use ----
    out_path = args.predictions_csv.replace("predictions_", "graded_")
    played.to_csv(out_path, index=False)
    print(f"Saved graded results to {out_path}")


if __name__ == "__main__":
    main()
