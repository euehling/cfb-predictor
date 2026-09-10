"""
Page 1 — Schedule. Season-long KPIs, the current week's schedule, and a
line chart tracking model vs. Vegas RMSE week over week.
"""

import os
import sys
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard_common import inject_css, discover_files, latest_week, styled_table, DATA_DIR

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


def compute_weekly_rmse(graded_files: list) -> pd.DataFrame:
    """One row per graded week: model RMSE and Vegas RMSE for that week
    specifically (not cumulative), for the trend chart."""
    rows = []
    for season, week, path in graded_files:
        df = pd.read_csv(path)
        model_rmse = np.sqrt((df["margin_error"] ** 2).mean())
        row = {"week_label": f"{season} Wk{week}", "season": season, "week": week, "Model": model_rmse}

        if "vegas_implied_margin" in df.columns:
            has_line = df["vegas_implied_margin"].notna()
            if has_line.sum() > 0:
                vegas_error = df.loc[has_line, "vegas_implied_margin"] - df.loc[has_line, "actual_margin"]
                row["Vegas"] = np.sqrt((vegas_error ** 2).mean())

        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["season", "week"])
    return result


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

    # ---- This week's schedule ----
    season, week, pred_path = latest_week(DATA_DIR)
    preds = pd.read_csv(pred_path)

    st.subheader(f"Week {week} Schedule")
    sched_cols = [c for c in ["home_team", "away_team", "start_date", "vegas_spread"] if c in preds.columns]
    schedule = preds[sched_cols].copy()
    if "start_date" in schedule.columns:
        schedule = schedule.sort_values("start_date")
    st.dataframe(styled_table(schedule), width='stretch', hide_index=True)
    st.divider()

    # ---- Weekly RMSE trend ----
    weekly = compute_weekly_rmse(graded_files)
    if len(weekly) >= 1:
        st.subheader("RMSE by week: Model vs. Vegas")
        value_cols = [c for c in ["Model", "Vegas"] if c in weekly.columns]
        long_df = weekly.melt(id_vars=["week_label", "season", "week"], value_vars=value_cols,
                               var_name="Series", value_name="RMSE")

        chart = alt.Chart(long_df).mark_line(point=True).encode(
            x=alt.X("week_label:N", sort=None, title=None,
                    axis=alt.Axis(labelColor="#b8b8c2", labelFontSize=11)),
            y=alt.Y("RMSE:Q", title="Margin RMSE (pts)",
                    axis=alt.Axis(labelColor="#b8b8c2", titleColor="#9db4f0")),
            color=alt.Color("Series:N", scale=alt.Scale(
                domain=["Model", "Vegas"], range=["#7fceac", "#9db4f0"]
            ), legend=alt.Legend(title=None, labelColor="#e8e8ec")),
            tooltip=["week_label", "Series", alt.Tooltip("RMSE:Q", format=".2f")],
        ).properties(height=320).configure_view(strokeWidth=0).configure(background="transparent")

        st.altair_chart(chart, width='stretch')


render()