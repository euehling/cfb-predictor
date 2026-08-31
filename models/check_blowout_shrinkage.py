"""
Tests whether the model systematically under-predicts margin specifically
in games Vegas expects to be lopsided — i.e. does model_edge_vs_vegas get
more negative as |vegas_spread| grows? This is the "does the model shrink
toward the middle on blowouts" question, tested across ALL games rather
than one anecdotal example.

Usage:
    python check_blowout_shrinkage.py ../data/predictions_2026_week1.csv
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else "../data/predictions_2026_week1.csv"
df = pd.read_csv(path)

df = df.dropna(subset=["vegas_spread", "model_edge_vs_vegas"])
df["vegas_favorite_margin"] = df["vegas_spread"].abs()  # size of the expected blowout, regardless of which side

print(f"Using {len(df)} games with both a Vegas line and a model prediction.\n")

# Bucket games by how lopsided Vegas expects them to be
bins = [0, 7, 14, 21, 100]
labels = ["Close (0-7)", "Moderate (7-14)", "Big favorite (14-21)", "Blowout (21+)"]
df["spread_bucket"] = pd.cut(df["vegas_favorite_margin"], bins=bins, labels=labels)

print("=== Mean model_edge_vs_vegas by how big the Vegas favorite is ===")
print("(Negative = model under-predicts the favorite's margin relative to Vegas)")
summary = df.groupby("spread_bucket", observed=True)["model_edge_vs_vegas"].agg(["mean", "std", "count"])
print(summary)
print()

# Direct correlation: does edge get more negative as the expected blowout grows?
r, p = stats.pearsonr(df["vegas_favorite_margin"], df["model_edge_vs_vegas"])
print(f"Correlation between |vegas_spread| and model_edge_vs_vegas: r={r:.3f} (p={p:.4f})")
print()
if r < -0.2 and p < 0.05:
    print("CONFIRMED PATTERN: the model systematically under-predicts more as the expected")
    print("blowout gets bigger. This is real shrinkage behavior, not one-game noise.")
elif abs(r) < 0.1:
    print("No meaningful pattern — the Alabama game looks like an isolated case, not a")
    print("systematic issue across this week's slate.")
else:
    print("Weak/inconclusive pattern — worth more data (more weeks) before concluding either way.")