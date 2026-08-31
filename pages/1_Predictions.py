"""
Page 2 — Predictions. The betting sheet: Vegas vs. model, suggested side,
confidence score, and Safe/Value pick badges.
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_common import (
    inject_css, latest_week, DATA_DIR, vegas_favorite, model_favorite,
    suggested_side, pick_badge_html, compute_confidence, confidence_label,
)

st.set_page_config(page_title="CFB Predictor — Predictions", layout="wide", page_icon="🎯")
inject_css()


def render():
    result = latest_week(DATA_DIR)
    if result is None:
        st.warning("No prediction files found yet. Run predict_week.py first.")
        return
    season, week, pred_path = result

    st.title(f"🎯 Predictions — {season} Week {week}")
    st.caption("predicted_margin and predicted scores are the MODEL'S OWN calls, not Vegas's. "
               "Vegas is shown for comparison.")

    preds = pd.read_csv(pred_path)

    preds["vegas_favorite"] = preds.apply(vegas_favorite, axis=1)
    preds["model_favorite"] = preds.apply(model_favorite, axis=1)
    preds["suggested_side"] = preds.apply(suggested_side, axis=1)
    preds["confidence_score"] = preds.apply(compute_confidence, axis=1)
    preds["confidence_label"] = preds["confidence_score"].apply(confidence_label)
    preds["badge"] = preds["model_edge_vs_vegas"].apply(pick_badge_html)

    display = preds[[
        "home_team", "away_team", "predicted_home_score", "predicted_away_score",
        "vegas_favorite", "model_favorite", "suggested_side", "badge",
        "confidence_score", "confidence_label",
    ]].copy()

    display["predicted_home_score"] = display["predicted_home_score"].round(0).astype(int)
    display["predicted_away_score"] = display["predicted_away_score"].round(0).astype(int)
    display["confidence_score"] = display["confidence_score"].round(1)

    display = display.rename(columns={
        "home_team": "Home", "away_team": "Away",
        "predicted_home_score": "Pred. Home", "predicted_away_score": "Pred. Away",
        "vegas_favorite": "Vegas", "model_favorite": "Model",
        "suggested_side": "Suggested Bet", "badge": "Pick",
        "confidence_score": "Conf.", "confidence_label": "",
    })

    html_table = display.to_html(escape=False, index=False, classes="styled-table", border=0)
    st.markdown(html_table, unsafe_allow_html=True)


render()