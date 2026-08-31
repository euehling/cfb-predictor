"""
Predicts one upcoming week of games using the saved models and
season-to-date features (power rating built only from games already
played this season — same leakage-free logic as training).

Usage:
    cd features
    python ../models/predict_week.py                 # auto-detects next unplayed week
    python ../models/predict_week.py --season 2026 --week 3

Requires CFBD_API_KEY set (pulls current-season games, SP+, talent,
rankings, and betting lines for the target week).
"""

import sys
import argparse
import json
import pandas as pd
import numpy as np
import xgboost as xgb

sys.path.insert(0, ".")
from cfbd_pull import CFBDClient, build_dynamic_tiers
from build_feature_table import (
    games_to_df, sp_to_df, talent_to_df, ap_ranks_by_week, locale_for_row, lines_to_df,
)
from power_rating import GameResult, Locale, score_game, TeamSeasonPower
from train import add_derived_features

MODELS_DIR = "../models"


def get_current_power_ratings(completed_games: pd.DataFrame, tiers_by_week: dict, ap_ranks_map: dict, season: int) -> dict:
    """
    Same accumulation logic as build_power_ratings_for_season, but we only
    need the FINAL state per team (their rating entering the next game),
    not a per-game snapshot history — this is for predicting a future
    game that hasn't happened yet, not backfilling training data.
    """
    completed_games = completed_games.sort_values("start_date")
    team_running = {}

    for _, row in completed_games.iterrows():
        week = row["week"]
        tiers = tiers_by_week.get((season, week), {})
        ap_ranks = ap_ranks_map.get((season, week), {})

        for team_col, opp_col, is_home in [("home_team", "away_team", True), ("away_team", "home_team", False)]:
            team = row[team_col]
            opp = row[opp_col]
            prior = team_running.get(team, TeamSeasonPower(team=team))

            opp_tier = tiers.get(opp, 3)
            opp_ap_rank = ap_ranks.get(opp)
            locale = locale_for_row(row, is_home)
            pts_for = row["home_points"] if is_home else row["away_points"]
            pts_against = row["away_points"] if is_home else row["home_points"]

            game_result = GameResult(
                opponent=opp, opponent_tier=opp_tier, opponent_ap_rank=opp_ap_rank,
                locale=locale, pts_for=pts_for, pts_against=pts_against,
            )
            gs = score_game(game_result)
            team_running[team] = TeamSeasonPower(team=team, games=list(prior.games) + [gs])

    return {team: tsp.power_rating for team, tsp in team_running.items()}


def load_models():
    with open(f"{MODELS_DIR}/feature_columns.json") as f:
        feature_cols = json.load(f)

    margin_model = xgb.XGBRegressor()
    margin_model.load_model(f"{MODELS_DIR}/margin_model.json")
    total_model = xgb.XGBRegressor()
    total_model.load_model(f"{MODELS_DIR}/total_model.json")
    win_model = xgb.XGBClassifier()
    win_model.load_model(f"{MODELS_DIR}/win_model.json")

    return margin_model, total_model, win_model, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, default=None,
                         help="If omitted, auto-detects the next week with unplayed games.")
    args = parser.parse_args()

    client = CFBDClient()
    season = args.season

    print(f"Pulling {season} season data...")
    all_games = client.get_games(season)
    all_sp = client.get_sp_ratings(season)
    all_talent = client.get_talent(season)
    all_rankings = client.get_ap_rankings(season)
    all_lines = client.get_betting_lines(season)

    games_df = pd.DataFrame([{
        "game_id": g.id, "season": g.season, "week": g.week,
        "season_type": g.season_type, "start_date": g.start_date,
        "home_team": g.home_team, "away_team": g.away_team,
        "home_points": g.home_points, "away_points": g.away_points,
        "neutral_site": g.neutral_site, "conference_game": g.conference_game,
    } for g in all_games])

    if games_df.empty:
        print(f"No games found for {season} yet — schedule may not be published.")
        return

    completed = games_df[games_df["home_points"].notna()]
    upcoming = games_df[games_df["home_points"].isna()]

    if args.week is not None:
        target_week = args.week
    else:
        if upcoming.empty:
            print(f"No upcoming games found for {season} — season may be complete.")
            return
        target_week = upcoming["week"].min()

    target_games = upcoming[upcoming["week"] == target_week]
    if target_games.empty:
        print(f"No upcoming games found for {season} week {target_week}.")
        return
    print(f"Predicting {len(target_games)} games for {season} week {int(target_week)}...\n")

    # SP+ / talent: prefer this season's data; if empty this early in the
    # season (SP+ in particular often isn't published until after week 1
    # or so), fall back to the PRIOR season's final ratings as a preseason
    # proxy — clearly worse than in-season SP+, but far better than
    # leaving every team at a null value.
    sp_df = sp_to_df(all_sp, season)
    talent_df = talent_to_df(all_talent, season)
    used_sp_fallback = sp_df.empty
    if used_sp_fallback:
        print(f"No {season} SP+ data yet — falling back to {season - 1} final SP+ as a preseason proxy.")
        prior_sp = client.get_sp_ratings(season - 1)
        sp_df = sp_to_df(prior_sp, season - 1)
        sp_df["season"] = season  # relabel so the merge below still lines up

    if talent_df.empty:
        print(f"No {season} talent data yet — falling back to {season - 1} talent as a proxy.")
        prior_talent = client.get_talent(season - 1)
        talent_df = talent_to_df(prior_talent, season - 1)
        talent_df["season"] = season

    ap_ranks_map = ap_ranks_by_week(all_rankings)

    # Tiers, built from whatever SP+ we have (current or fallback)
    tiers_by_week = {}
    weeks_so_far = completed["week"].dropna().unique()
    sp_list_for_tiers = all_sp if not used_sp_fallback else client.get_sp_ratings(season - 1)
    for week in weeks_so_far:
        ap_this_week = ap_ranks_map.get((season, week), {})
        tiers_by_week[(season, week)] = build_dynamic_tiers(sp_list_for_tiers, ap_this_week)
    # Also compute one tier snapshot for the target week itself (used to
    # score the upcoming games' opponents, even though those games haven't
    # been played — tiers describe team quality, not game outcomes).
    ap_target_week = ap_ranks_map.get((season, target_week), {})
    tiers_by_week[(season, target_week)] = build_dynamic_tiers(sp_list_for_tiers, ap_target_week)

    power_ratings = get_current_power_ratings(completed, tiers_by_week, ap_ranks_map, season)

    rows = []
    for _, g in target_games.iterrows():
        home, away = g["home_team"], g["away_team"]
        row = {
            "game_id": g["game_id"], "start_date": g["start_date"],
            "home_team": home, "away_team": away,
            "neutral_site": int(g["neutral_site"]), "conference_game": int(g["conference_game"]),
            "home_power_rating": power_ratings.get(home, 0.0),
            "away_power_rating": power_ratings.get(away, 0.0),
        }
        rows.append(row)
    feat_df = pd.DataFrame(rows)

    sp_home = sp_df.rename(columns={"team": "home_team", "sp_rating": "home_sp_rating",
                                     "sp_offense": "home_sp_offense", "sp_defense": "home_sp_defense"})
    sp_away = sp_df.rename(columns={"team": "away_team", "sp_rating": "away_sp_rating",
                                     "sp_offense": "away_sp_offense", "sp_defense": "away_sp_defense"})
    feat_df = feat_df.merge(sp_home[["home_team", "home_sp_rating", "home_sp_offense", "home_sp_defense"]],
                             on="home_team", how="left")
    feat_df = feat_df.merge(sp_away[["away_team", "away_sp_rating", "away_sp_offense", "away_sp_defense"]],
                             on="away_team", how="left")

    talent_home = talent_df.rename(columns={"team": "home_team", "talent": "home_talent"})
    talent_away = talent_df.rename(columns={"team": "away_team", "talent": "away_talent"})
    feat_df = feat_df.merge(talent_home[["home_team", "home_talent"]], on="home_team", how="left")
    feat_df = feat_df.merge(talent_away[["away_team", "away_talent"]], on="away_team", how="left")

    feat_df = add_derived_features(feat_df)

    margin_model, total_model, win_model, feature_cols = load_models()
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        raise SystemExit(f"Feature mismatch — missing columns the saved model expects: {missing}")

    X = feat_df[feature_cols]
    margin_pred = margin_model.predict(X)
    total_pred = total_model.predict(X)
    win_prob = win_model.predict_proba(X)[:, 1]

    feat_df["predicted_margin"] = margin_pred.round(1)
    feat_df["predicted_total"] = total_pred.round(1)
    feat_df["predicted_home_score"] = ((total_pred + margin_pred) / 2).round(0)
    feat_df["predicted_away_score"] = ((total_pred - margin_pred) / 2).round(0)
    feat_df["home_win_prob"] = win_prob.round(3)

    lines_df = lines_to_df(all_lines)
    if not lines_df.empty:
        feat_df = feat_df.merge(lines_df, on="game_id", how="left")
        feat_df["vegas_implied_margin"] = -feat_df["vegas_spread"]
        feat_df["model_edge_vs_vegas"] = (feat_df["predicted_margin"] - feat_df["vegas_implied_margin"]).round(1)

    display_cols = ["home_team", "away_team", "predicted_home_score", "predicted_away_score",
                     "predicted_margin", "home_win_prob"]
    if "vegas_spread" in feat_df.columns:
        display_cols += ["vegas_spread", "model_edge_vs_vegas"]

    print(feat_df[display_cols].to_string(index=False))

    out_path = f"../data/predictions_{season}_week{int(target_week)}.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()