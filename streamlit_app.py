"""
Page 1 — Schedule. Season-long KPIs, the week's schedule, and a live
results table that re-pulls from CFBD on every refresh (cached briefly
so rapid refreshes don't hammer the API).
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard_common import (
    inject_css, discover_files, latest_week, styled_table, DATA_DIR,
    get_cfbd_api_key,
)

st.set_page_config(page_title="CFB Predictor — Schedule", layout="wide", page_icon="🏈")
inject_css()


def compute_season_summary(graded_files: list) -> dict:
    all_graded = []
    for season, week, path in graded_files:
        df = pd.read_csv(path)
        df["season"] = season
        df["week"] = week
        all_graded.append(df)

    if not all_graded:
        return {}

    combined = pd.concat(all_graded, ignore_index=True)
    margin_rmse = np.sqrt((combined["margin_error"] ** 2).mean())
    win_acc = combined["win_correct"].mean()

    summary = {"games_graded": len(combined), "margin_rmse": margin_rmse, "win_accuracy": win_acc}

    if "vegas_implied_margin" in combined.columns:
        has_line = combined["vegas_implied_margin"].notna()
        if has_line.sum() > 0:
            vegas_error = combined.loc[has_line, "vegas_implied_margin"] - combined.loc[has_line, "actual_margin"]
            summary["vegas_rmse"] = np.sqrt((vegas_error ** 2).mean())

    return summary


@st.cache_data(ttl=300)
def pull_live_scores(season: int, game_ids: tuple):
    """Live-pulls current scores from CFBD for the given game_ids.
    Cached 5 minutes so repeated manual refreshes in a short window don't
    re-hit the API unnecessarily, while staying fresh across a normal
    day of checking in."""
    from cfbd_pull import CFBDClient
    api_key = get_cfbd_api_key()
    client = CFBDClient(api_key=api_key)
    games = client.get_games(season)
    rows = [{"game_id": g.id, "actual_home_points": g.home_points, "actual_away_points": g.away_points}
            for g in games if g.id in game_ids]
    return pd.DataFrame(rows)


def render():
    st.title("📅 Schedule")

    pred_files = discover_files(DATA_DIR, "predictions")
    graded_files = discover_files(DATA_DIR, "graded")

    if not pred_files:
        st.warning("No prediction files found yet. Run predict_week.py first.")
        return

    # ---- Season-long KPIs ----
    summary = compute_season_summary(graded_files)
    if summary:
        st.subheader("Season so far")
        cols = st.columns(4)
        cols[0].metric("Games graded", summary["games_graded"])
        cols[1].metric("Margin RMSE", f"{summary['margin_rmse']:.2f} pts")
        cols[2].metric("Win/loss accuracy", f"{summary['win_accuracy']:.1%}")
        if "vegas_rmse" in summary:
            diff = summary["vegas_rmse"] - summary["margin_rmse"]
            label = "Model ahead of Vegas" if diff > 0 else "Vegas ahead of model"
            cols[3].metric(label, f"{abs(diff):.2f} pts RMSE", delta=f"{diff:+.2f}")
        st.divider()

    season, week, pred_path = latest_week(DATA_DIR)
    preds = pd.read_csv(pred_path)

    st.subheader(f"Week {week} Schedule")
    sched_cols = [c for c in ["home_team", "away_team", "start_date", "vegas_spread"] if c in preds.columns]
    schedule = preds[sched_cols].copy()
    if "start_date" in schedule.columns:
        schedule = schedule.sort_values("start_date")
    st.dataframe(styled_table(schedule), width='stretch', hide_index=True)

    st.subheader("Live Results")
    st.caption("Refresh the page any time to pull current scores. Updates as games finish.")

    if "game_id" not in preds.columns:
        st.info("No game_id column found — can't pull live scores for this file.")
        return

    api_key = get_cfbd_api_key()
    if not api_key:
        st.warning("No CFBD_API_KEY found (set it as an env var locally, or a Streamlit secret when deployed) — can't pull live scores.")
        return

    try:
        live = pull_live_scores(season, tuple(preds["game_id"].tolist()))
    except Exception as e:
        st.error(f"Couldn't pull live scores: {e}")
        return

    merged = preds.merge(live, on="game_id", how="left")
    merged["spread"] = merged["actual_home_points"] - merged["actual_away_points"]

    live_cols = [c for c in ["home_team", "away_team", "actual_home_points", "actual_away_points", "spread"]
                 if c in merged.columns]
    live_display = merged[live_cols]
    st.dataframe(styled_table(live_display, edge_cols=[]), width='stretch', hide_index=True)


render()