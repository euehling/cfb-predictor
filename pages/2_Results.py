"""
Page 3 — Results. Vegas vs. model vs. actual, side by side, plus
best/worst calls for whichever week you pick.
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_common import inject_css, discover_files, styled_table, DATA_DIR

st.set_page_config(page_title="CFB Predictor — Results", layout="wide", page_icon="📊")
inject_css()


def render():
    st.title("📊 Results")

    graded_files = discover_files(DATA_DIR, "graded")
    if not graded_files:
        st.info("No weeks have been graded yet — run grade_predictions.py once games finish.")
        return

    week_labels = [f"{s} Week {w}" for s, w, _ in graded_files]
    selected = st.selectbox("Select week", week_labels, index=len(week_labels) - 1)
    idx = week_labels.index(selected)
    season, week, path = graded_files[idx]

    graded = pd.read_csv(path)

    st.subheader(f"{season} Week {week}: Vegas vs. Model vs. Actual")

    cols = st.columns(3)
    cols[0].metric("Games graded", len(graded))
    import numpy as np
    margin_rmse = np.sqrt((graded["margin_error"] ** 2).mean())
    cols[1].metric("Model margin RMSE", f"{margin_rmse:.2f} pts")
    if "vegas_implied_margin" in graded.columns:
        has_line = graded["vegas_implied_margin"].notna()
        if has_line.sum() > 0:
            vegas_error = graded.loc[has_line, "vegas_implied_margin"] - graded.loc[has_line, "actual_margin"]
            vegas_rmse = np.sqrt((vegas_error ** 2).mean())
            cols[2].metric("Vegas margin RMSE", f"{vegas_rmse:.2f} pts")

    compare_cols = [c for c in [
        "home_team", "away_team", "vegas_implied_margin", "predicted_margin",
        "actual_margin", "margin_error",
    ] if c in graded.columns]
    st.dataframe(styled_table(graded[compare_cols]), width='stretch', hide_index=True)

    st.subheader("Points comparison")
    score_cols = [c for c in [
        "home_team", "away_team", "predicted_home_score", "actual_home_points",
        "predicted_away_score", "actual_away_points",
    ] if c in graded.columns]
    st.dataframe(styled_table(graded[score_cols], edge_cols=[]), width='stretch', hide_index=True)

    st.subheader("Best vs. worst predictions")
    tab1, tab2 = st.tabs(["Best", "Worst"])
    result_cols = [c for c in ["home_team", "away_team", "predicted_margin", "actual_margin", "margin_error"]
                   if c in graded.columns]
    with tab1:
        st.dataframe(styled_table(graded.nsmallest(5, "abs_margin_error")[result_cols]), hide_index=True)
    with tab2:
        st.dataframe(styled_table(graded.nlargest(5, "abs_margin_error")[result_cols]), hide_index=True)


render()