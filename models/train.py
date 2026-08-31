"""
Trains the CFB prediction models: margin regression, total-points
regression, and a home-win probability classifier. Time-based split
(never random — see README) with recency weighting, benchmarked against
the Vegas closing line.

Usage:
    cd features
    python ../models/train.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
import json
import os

DATA_PATH = "../data/training_table.csv"
MODELS_DIR = "../models"

# Columns that exist in the table but are NOT model inputs — identifiers,
# raw labels, targets, or benchmark-only data (Vegas lines — see
# lines_to_df() in build_feature_table.py for why those stay out).
NON_FEATURE_COLUMNS = {
    "game_id", "season", "week", "season_type", "start_date",
    "home_team", "away_team", "home_points", "away_points",
    "margin", "total_points", "home_win", "sample_weight",
    "vegas_spread", "vegas_over_under", "vegas_num_books",
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered diff features — home minus away for each paired stat.
    XGBoost can learn interactions on its own, but explicit diffs give
    it a head start and make feature importances easier to read.

    Also adds explicit missing-data flags for SP+/talent. These matter
    because SP+ and the talent composite are FBS-only metrics — when an
    FBS team plays a D2 or FCS opponent (legitimate scheduling, not a
    data error), that opponent's SP+/talent come back NaN. XGBoost
    handles NaN natively, but a NaN alone doesn't tell the model WHY the
    value is missing — the flag gives it a direct, explicit signal it
    can learn "this usually means a lopsided game" from, rather than
    inferring it indirectly and under-predicting the resulting blowout
    (confirmed pattern: model_edge_vs_vegas correlates at r=-0.60 with
    Vegas spread size on 2026 week-1 predictions, worst in exactly this
    missing-data scenario)."""
    df = df.copy()
    df["power_diff"] = df["home_power_rating"] - df["away_power_rating"]
    df["sp_diff"] = df["home_sp_rating"] - df["away_sp_rating"]
    df["talent_diff"] = df["home_talent"] - df["away_talent"]
    df["neutral_site"] = df["neutral_site"].astype(int)
    df["conference_game"] = df["conference_game"].astype(int)

    df["home_sp_missing"] = df["home_sp_rating"].isna().astype(int)
    df["away_sp_missing"] = df["away_sp_rating"].isna().astype(int)
    df["home_talent_missing"] = df["home_talent"].isna().astype(int)
    df["away_talent_missing"] = df["away_talent"].isna().astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def time_based_split(df: pd.DataFrame):
    train = df[df["season"].isin([2022, 2023])]
    val = df[df["season"] == 2024]
    test = df[df["season"] == 2025]
    return train, val, test


def train_regressor(X_train, y_train, w_train, X_val, y_val, label: str) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train, sample_weight=w_train,
        eval_set=[(X_val, y_val)], verbose=False,
    )
    print(f"[{label}] best iteration: {model.best_iteration}, "
          f"best val RMSE: {model.best_score:.3f}")
    return model


def train_classifier(X_train, y_train, w_train, X_val, y_val) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        early_stopping_rounds=30,
        eval_metric="logloss",
    )
    model.fit(
        X_train, y_train, sample_weight=w_train,
        eval_set=[(X_val, y_val)], verbose=False,
    )
    print(f"[home_win] best iteration: {model.best_iteration}, "
          f"best val logloss: {model.best_score:.3f}")
    return model


def evaluate_vs_vegas(test_df: pd.DataFrame, predicted_margin: np.ndarray):
    """
    Vegas spread convention: negative = home favored. So Vegas's implied
    predicted margin for the home team is -vegas_spread. Only scored on
    test rows that actually have a line (not every game gets one, e.g.
    small non-conference games).
    """
    if "vegas_spread" not in test_df.columns:
        print("\nNo vegas_spread column found in the data — the training "
              "table wasn't built with betting lines. Rerun "
              "build_feature_table.py to regenerate it, then retrain.")
        return

    has_line = test_df["vegas_spread"].notna()
    n = has_line.sum()
    if n == 0:
        print("No Vegas lines available in test set — skipping benchmark comparison.")
        return

    actual_margin = test_df.loc[has_line, "margin"].values
    vegas_implied_margin = -test_df.loc[has_line, "vegas_spread"].values
    model_margin = predicted_margin[has_line.values]

    model_rmse = np.sqrt(mean_squared_error(actual_margin, model_margin))
    vegas_rmse = np.sqrt(mean_squared_error(actual_margin, vegas_implied_margin))

    print(f"\n=== Model vs. Vegas on {n} test games with lines available ===")
    print(f"Model margin RMSE:  {model_rmse:.2f} points")
    print(f"Vegas margin RMSE:  {vegas_rmse:.2f} points")
    if model_rmse < vegas_rmse:
        print(f"Model beat Vegas by {vegas_rmse - model_rmse:.2f} points RMSE.")
    else:
        print(f"Vegas beat the model by {model_rmse - vegas_rmse:.2f} points RMSE.")


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_derived_features(df)
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} features: {feature_cols}\n")

    train_df, val_df, test_df = time_based_split(df)
    print(f"Train: {len(train_df)} games (2022-2023)")
    print(f"Val:   {len(val_df)} games (2024)")
    print(f"Test:  {len(test_df)} games (2025)\n")

    X_train, X_val, X_test = train_df[feature_cols], val_df[feature_cols], test_df[feature_cols]
    w_train = train_df["sample_weight"]

    # ---- Margin regression ----
    margin_model = train_regressor(
        X_train, train_df["margin"], w_train, X_val, val_df["margin"], "margin"
    )
    margin_pred = margin_model.predict(X_test)
    margin_rmse = np.sqrt(mean_squared_error(test_df["margin"], margin_pred))
    print(f"Margin RMSE on 2025 test set: {margin_rmse:.2f} points\n")

    # ---- Total points regression ----
    total_model = train_regressor(
        X_train, train_df["total_points"], w_train, X_val, val_df["total_points"], "total_points"
    )
    total_pred = total_model.predict(X_test)
    total_rmse = np.sqrt(mean_squared_error(test_df["total_points"], total_pred))
    print(f"Total points RMSE on 2025 test set: {total_rmse:.2f} points\n")

    # ---- Home win classifier ----
    win_model = train_classifier(
        X_train, train_df["home_win"], w_train, X_val, val_df["home_win"]
    )
    win_pred_proba = win_model.predict_proba(X_test)[:, 1]
    win_pred = (win_pred_proba >= 0.5).astype(int)
    win_acc = accuracy_score(test_df["home_win"], win_pred)
    win_logloss = log_loss(test_df["home_win"], win_pred_proba)
    print(f"Home-win accuracy on 2025 test set: {win_acc:.3f}")
    print(f"Home-win log loss on 2025 test set: {win_logloss:.3f}\n")

    # ---- Derive individual scores from margin + total ----
    home_pred = (total_pred + margin_pred) / 2
    away_pred = (total_pred - margin_pred) / 2
    home_score_rmse = np.sqrt(mean_squared_error(test_df["home_points"], home_pred))
    away_score_rmse = np.sqrt(mean_squared_error(test_df["away_points"], away_pred))
    print(f"Derived home score RMSE: {home_score_rmse:.2f} points")
    print(f"Derived away score RMSE: {away_score_rmse:.2f} points\n")

    # ---- Benchmark margin predictions against Vegas ----
    evaluate_vs_vegas(test_df, margin_pred)

    # ---- Feature importance (margin model) ----
    print("\n=== Top 10 features by importance (margin model) ===")
    importances = pd.Series(margin_model.feature_importances_, index=feature_cols)
    print(importances.sort_values(ascending=False).head(10))

    # ---- Save models ----
    os.makedirs(MODELS_DIR, exist_ok=True)
    margin_model.save_model(f"{MODELS_DIR}/margin_model.json")
    total_model.save_model(f"{MODELS_DIR}/total_model.json")
    win_model.save_model(f"{MODELS_DIR}/win_model.json")
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()