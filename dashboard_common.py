"""
Shared code for all three dashboard pages: CSS injection, number
formatting, confidence scoring, and the Vegas/model betting-side helpers.
Import from streamlit_app.py and pages/*.py — don't duplicate this logic
in each page file.
"""

import os
import re
import glob
import sys
import pandas as pd
import numpy as np
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# so we can import CFBDClient for the live-refresh table on Page 1
sys.path.insert(0, os.path.join(REPO_ROOT, "features"))


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@400;500;600&display=swap');

.stApp {
    background: #0a0a0c;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #f5f3ee !important;
    font-weight: 800 !important;
}

p, div, span, label {
    font-family: 'Inter', sans-serif;
    color: #b8b8c2;
}

.section-label {
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #7fceac;
    margin-bottom: 4px;
}

[data-testid="stMetric"] {
    background: #14141a;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 16px;
}

[data-testid="stMetricValue"] {
    color: #f5f3ee !important;
    font-family: 'Playfair Display', serif !important;
}

[data-testid="stMetricLabel"] {
    color: #9db4f0 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.75rem !important;
}

.stDataFrame, [data-testid="stTable"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    overflow: hidden;
}

.pill-badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 20px;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-safe {
    background: rgba(127,206,172,0.12);
    color: #7fceac;
    border: 1px solid rgba(127,206,172,0.4);
}
.badge-value {
    background: rgba(230,167,95,0.12);
    color: #e6a75f;
    border: 1px solid rgba(230,167,95,0.4);
}
.badge-neutral {
    background: rgba(157,180,240,0.10);
    color: #9db4f0;
    border: 1px solid rgba(157,180,240,0.35);
}

.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
}
.styled-table th {
    color: #9db4f0;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.72rem;
    font-weight: 600;
}
.styled-table td {
    padding: 9px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #e8e8ec;
}
.styled-table tr:hover td {
    background: rgba(255,255,255,0.02);
}
.edge-pos { color: #7fceac; }
.edge-neg { color: #e08585; }

/* Multiselect filter tags — override Streamlit's default red */
[data-baseweb="tag"] {
    background-color: rgba(167, 139, 250, 0.18) !important;
    border: 1px solid rgba(167, 139, 250, 0.5) !important;
}
[data-baseweb="tag"] span {
    color: #c4b5fd !important;
}
[data-baseweb="tag"] svg {
    fill: #c4b5fd !important;
}

hr {
    border-color: rgba(255,255,255,0.08) !important;
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def get_cfbd_api_key():
    """Checks Streamlit secrets first (for cloud deployment), then falls
    back to the environment variable (for local use)."""
    try:
        if "CFBD_API_KEY" in st.secrets:
            return st.secrets["CFBD_API_KEY"]
    except Exception:
        pass
    return os.environ.get("CFBD_API_KEY")


def discover_files(data_dir: str, prefix: str) -> list:
    """Returns [(season, week, path), ...] sorted by season then week,
    for files matching {prefix}_{season}_week{week}.csv."""
    pattern = os.path.join(data_dir, f"{prefix}_*_week*.csv")
    results = []
    for path in glob.glob(pattern):
        match = re.search(rf"{prefix}_(\d{{4}})_week(\d+)", os.path.basename(path))
        if match:
            results.append((int(match.group(1)), int(match.group(2)), path))
    return sorted(results)


def latest_week(data_dir: str, prefix: str = "predictions"):
    """Returns (season, week, path) for the most recent predictions file,
    or None if none exist."""
    files = discover_files(data_dir, prefix)
    return files[-1] if files else None


# ---- Number formatting — explicit format STRINGS, not just rounded
# values, since Styler needs an actual format spec to control what's
# rendered (rounding the value alone doesn't stop pandas from padding
# zeros back on when it displays a float). ----
FORMAT_RULES = {
    "predicted_home_score": "{:.0f}",
    "predicted_away_score": "{:.0f}",
    "actual_home_points": "{:.0f}",
    "actual_away_points": "{:.0f}",
    "predicted_margin": "{:.2f}",
    "actual_margin": "{:.2f}",
    "margin_error": "{:.2f}",
    "abs_margin_error": "{:.2f}",
    "home_win_prob": "{:.2f}",
    "vegas_spread": "{:.2f}",
    "vegas_over_under": "{:.2f}",
    "vegas_implied_margin": "{:.2f}",
    "model_edge_vs_vegas": "{:.2f}",
    "confidence_score": "{:.1f}",
    "spread": "{:.1f}",
}


def styled_table(df: pd.DataFrame, edge_cols=None):
    """Returns a Styler with correct per-column decimal formatting and
    green/red coloring on any edge/error columns. edge_cols defaults to
    model_edge_vs_vegas and margin_error if present."""
    if edge_cols is None:
        edge_cols = [c for c in ["model_edge_vs_vegas", "margin_error"] if c in df.columns]

    rules = {k: v for k, v in FORMAT_RULES.items() if k in df.columns}
    styled = df.style.format(rules, na_rep="—")

    def color_edge(val):
        if pd.isna(val):
            return ""
        return "color: #7fceac" if val > 0 else "color: #e08585"

    if edge_cols:
        styled = styled.map(color_edge, subset=edge_cols)
    return styled


# ---- Betting-side helpers ----

def vegas_favorite(row) -> str:
    spread = row.get("vegas_spread")
    if pd.isna(spread):
        return "—"
    if spread == 0:
        return "Pick'em"
    if spread < 0:
        return f"{row['home_team']} {spread:.2f}"
    return f"{row['away_team']} {-spread:.2f}"


def model_favorite(row) -> str:
    margin = row.get("predicted_margin")
    if pd.isna(margin):
        return "—"
    if margin == 0:
        return "Pick'em"
    if margin > 0:
        return f"{row['home_team']} -{margin:.2f}"
    return f"{row['away_team']} -{abs(margin):.2f}"


def suggested_side(row) -> str:
    """Which side to bet AT VEGAS'S NUMBER, based on which way the model
    disagrees with the market. Positive edge = model thinks the home team
    covers; negative edge = model thinks the away team covers (getting
    the points)."""
    edge = row.get("model_edge_vs_vegas")
    spread = row.get("vegas_spread")
    if pd.isna(edge) or pd.isna(spread):
        return "—"
    if edge == 0:
        return "No edge"
    if edge > 0:
        return f"{row['home_team']} {spread:.2f}"
    away_line = -spread
    sign = "+" if away_line >= 0 else ""
    return f"{row['away_team']} {sign}{away_line:.2f}"


def pick_badge_html(edge, safe_threshold=3.0, value_threshold=7.0) -> str:
    if pd.isna(edge):
        return ""
    a = abs(edge)
    if a <= safe_threshold:
        return '<span class="pill-badge badge-safe">SAFE</span>'
    if a >= value_threshold:
        return '<span class="pill-badge badge-value">VALUE</span>'
    return '<span class="pill-badge badge-neutral">MODERATE</span>'


# ---- Confidence score ----
# Built from win-probability decisiveness, then explicitly discounted in
# the exact situations we proved are less reliable: early season (power
# ratings still at 0, confirmed r=-0.34 to -0.61 shrinkage correlation),
# blowout-range predictions (21+ points, confirmed shrinkage territory),
# and missing opponent data (FCS/D2 mismatches, confirmed underprediction
# pattern). This is NOT just "how far win_prob is from 50%" — it's that,
# discounted by what we actually learned building this model.

def compute_confidence(row) -> float:
    prob = row.get("home_win_prob", 0.5)
    if pd.isna(prob):
        prob = 0.5
    base = abs(prob - 0.5) * 2  # 0 to 1

    penalty = 1.0

    hp = row.get("home_power_rating")
    ap = row.get("away_power_rating")
    if pd.notna(hp) and pd.notna(ap) and hp == 0 and ap == 0:
        penalty *= 0.6  # early season — no in-season signal yet

    margin = row.get("predicted_margin")
    if pd.notna(margin) and abs(margin) >= 21:
        penalty *= 0.75  # blowout range — confirmed shrinkage territory

    missing_flags = ["home_sp_missing", "away_sp_missing", "home_talent_missing", "away_talent_missing"]
    if any(row.get(f) == 1 for f in missing_flags if f in row.index):
        penalty *= 0.7  # FCS/D2 opponent — confirmed blind spot

    return round(base * penalty * 100, 1)


def confidence_label(score: float) -> str:
    if score >= 60:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


GLOSSARY = [
    ("predicted_home_score / predicted_away_score",
     "The model's predicted final score for each team."),
    ("predicted_margin",
     "This is the MODEL'S OWN predicted point margin for the HOME team — not Vegas's number. Positive means the model favors the home team; negative means it favors the away team."),
    ("home_win_prob",
     "The model's estimated probability the home team wins, 0 to 1."),
    ("vegas_spread",
     "The closing betting-market spread. Negative means the HOME team is favored by that many points (e.g. -7.0 = home favored by 7). Positive means the away team is favored."),
    ("Vegas / Model (Predictions page)",
     "Vegas is the market's favorite and number. Model is the model's own favorite and number. Compare them directly — when they disagree, that's where the Suggested Bet comes from."),
    ("Suggested Bet",
     "Whichever side the model thinks will cover VEGAS'S number, based on which way the model and the market disagree. If the model's margin is bigger than Vegas's, it suggests the home team; if smaller, it suggests the away team getting the points."),
    ("model_edge_vs_vegas",
     "How much bigger the model's predicted home margin is than what Vegas implies. Positive = model favors the home team more than Vegas does. Negative = model favors the home team less (or favors the away team more)."),
    ("SAFE / VALUE / MODERATE badge",
     "SAFE = model and Vegas are close together (small edge) — lower risk the model is the outlier. VALUE = model and Vegas diverge a lot (big edge) — the classic 'the market may be wrong' bet. MODERATE = in between."),
    ("Confidence score",
     "A 0-100 score for how much to trust this specific prediction. It starts from how decisive the win probability is, then gets DISCOUNTED for situations we specifically proved are less reliable: early season (power ratings still at 0 for both teams — confirmed the model's predictions are shakiest here), blowout-range predictions (21+ point margins — confirmed the model tends to under-call these), and opponent data gaps (FCS/D2 opponents with no SP+/talent data — confirmed underprediction pattern). A whole week showing low scores usually just means it's early season, not that something's broken."),
    ("actual_margin",
     "The real final point margin for the home team, once the game has been played."),
    ("margin_error",
     "predicted_margin minus actual_margin — how far off the model was. Positive means it overestimated the home team; negative means it underestimated them."),
    ("RMSE (Root Mean Squared Error)",
     "The main accuracy score used throughout — roughly, the typical size (in points) of the model's prediction errors. Lower is better."),
    ("win/loss accuracy",
     "Percentage of games where the model correctly predicted which team would win, regardless of margin."),
    ("home_power_rating / away_power_rating",
     "Each team's power rating entering this game, from the custom formula (win points, road/MOV bonuses, tier multipliers, strength of schedule). Starts at 0 every season and builds up as games are played."),
    ("sp_rating",
     "Bill Connelly's SP+ — a predictive, opponent-adjusted efficiency rating built from play-by-play data, not just final scores."),
    ("talent",
     "CFBD's team talent composite — a recruiting-based measure of roster talent, independent of how the season is going."),
]