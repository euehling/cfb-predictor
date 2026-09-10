"""
Page 2 — Predictions. The betting sheet: Vegas vs. model, suggested side,
confidence score, Safe/Value pick badges, kickoff time in Central Time,
and filters by Pick and Confidence.
"""

import os
import sys
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_common import (
    inject_css, latest_week, DATA_DIR, vegas_favorite, model_favorite,
    suggested_side, pick_badge_html, compute_confidence, confidence_label,
)

st.set_page_config(page_title="CFB Predictor — Predictions", layout="wide", page_icon="🎯")
inject_css()

CENTRAL = ZoneInfo("America/Chicago")


def badge_text(edge, safe_threshold=3.0, value_threshold=7.0) -> str:
    """Plain-text version of pick_badge_html's logic, for filtering — the
    HTML version can't be filtered on directly since it's markup, not a
    clean category."""
    if pd.isna(edge):
        return ""
    a = abs(edge)
    if a <= safe_threshold:
        return "SAFE"
    if a >= value_threshold:
        return "VALUE"
    return "MODERATE"


def render():
    result = latest_week(DATA_DIR)
    if result is None:
        st.warning("No prediction files found yet. Run predict_week.py first.")
        return
    season, week, pred_path = result

    st.title(f"🎯 Predictions — {season} Week {week}")
    st.caption(
        "Predicted scores are the model's own calls. **Pick** shows how far the model's "
        "number is from Vegas's (Safe = close agreement, Value = big gap, Moderate = in "
        "between). **Confidence** (0-100, Low/Medium/High) shows how much to trust this "
        "specific prediction, independent of Vegas."
    )

    preds = pd.read_csv(pred_path)

    preds["vegas_favorite"] = preds.apply(vegas_favorite, axis=1)
    preds["model_favorite"] = preds.apply(model_favorite, axis=1)
    preds["suggested_side"] = preds.apply(suggested_side, axis=1)
    preds["confidence_score"] = preds.apply(compute_confidence, axis=1)
    preds["confidence_label"] = preds["confidence_score"].apply(confidence_label)
    preds["pick_text"] = preds["model_edge_vs_vegas"].apply(badge_text)
    preds["badge"] = preds["model_edge_vs_vegas"].apply(pick_badge_html)

    # ---- Kickoff time, converted to Central and sorted earliest-first ----
    if "start_date" in preds.columns:
        parsed = pd.to_datetime(preds["start_date"], utc=True)
        preds["kickoff_ct"] = parsed.dt.tz_convert(CENTRAL)
        preds = preds.sort_values("kickoff_ct")
        preds["kickoff_display"] = preds["kickoff_ct"].dt.strftime("%a %b %d, %I:%M %p CT")
    else:
        preds["kickoff_display"] = ""

    # ---- Filters ----
    filter_cols = st.columns(3)
    pick_options = ["SAFE", "MODERATE", "VALUE"]
    selected_picks = filter_cols[0].multiselect("Filter by Pick", pick_options, default=pick_options)

    conf_label_options = ["Low", "Medium", "High"]
    selected_conf_labels = filter_cols[1].multiselect(
        "Filter by Confidence level", conf_label_options, default=conf_label_options
    )

    conf_range = filter_cols[2].slider("Confidence score range", 0, 100, (0, 100))

    filtered = preds[
        preds["pick_text"].isin(selected_picks)
        & preds["confidence_label"].isin(selected_conf_labels)
        & preds["confidence_score"].between(conf_range[0], conf_range[1])
    ]

    display = filtered[[
        "kickoff_display", "home_team", "away_team", "predicted_home_score", "predicted_away_score",
        "vegas_favorite", "model_favorite", "suggested_side", "badge",
        "confidence_score", "confidence_label",
    ]].copy()

    display["predicted_home_score"] = display["predicted_home_score"].round(0).astype(int)
    display["predicted_away_score"] = display["predicted_away_score"].round(0).astype(int)
    display["confidence_score"] = display["confidence_score"].round(1)

    display = display.rename(columns={
        "kickoff_display": "Date/Time (CT)",
        "home_team": "Home", "away_team": "Away",
        "predicted_home_score": "Pred. Home", "predicted_away_score": "Pred. Away",
        "vegas_favorite": "Vegas", "model_favorite": "Model",
        "suggested_side": "Suggested Bet", "badge": "Pick",
        "confidence_score": "Conf.", "confidence_label": "",
    })

    if display.empty:
        st.info("No games match the current filters.")
        return

    html_table = display.to_html(escape=False, index=False, classes="styled-table", border=0)
    st.markdown(html_table, unsafe_allow_html=True)


render()