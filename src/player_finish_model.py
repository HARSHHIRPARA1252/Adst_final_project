"""Leakage-safe player finish regression for BMSD MVP-style forecasts.

This predicts a player's match-level finish count. The source does not expose
per-match MVP award labels, so the app uses predicted finishes to identify an
MVP-style candidate and does not report an award probability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .data_preprocessing import canonical_team_key
except ImportError:
    from data_preprocessing import canonical_team_key


PLAYER_FEATURES = [
    "matches_before",
    "prev_avg_finishes",
    "recent3_avg_finishes",
    "prev_std_finishes",
    "team_prev_avg_finishes",
]
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "bmsd_player_finishes_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "bmsd_player_finish_feature_names.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_results" / "player_finish_model_comparison.csv"
METADATA_PATH = PROJECT_ROOT / "reports" / "model_results" / "player_finish_model_metadata.json"


def _matched_player_rows(player_matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    required_players = {"player_id", "player", "team", "source_match_id", "elims"}
    required_fixtures = {"source_match_id", "overall_match", "status"}
    if player_matches.empty or fixtures.empty or not required_players.issubset(player_matches.columns) or not required_fixtures.issubset(fixtures.columns):
        return pd.DataFrame()
    source = fixtures.copy()
    source["overall_match"] = pd.to_numeric(source["overall_match"], errors="coerce")
    source = source[source["status"].fillna("").astype(str).str.upper().eq("COMPLETED")]
    source = source.dropna(subset=["overall_match"]).groupby("source_match_id", as_index=False)["overall_match"].max()
    rows = player_matches.merge(source, on="source_match_id", how="inner")
    rows["finishes"] = pd.to_numeric(rows["elims"], errors="coerce")
    rows = rows.dropna(subset=["finishes", "overall_match"])
    rows = rows[rows["finishes"].ge(0)].copy()
    rows["player_id"] = rows["player_id"].fillna("").astype(str)
    rows["team_key"] = rows["team"].map(canonical_team_key)
    rows = rows[rows["player_id"].str.strip().ne("") & rows["team_key"].ne("")]
    return rows.drop_duplicates(["player_id", "source_match_id"], keep="last").sort_values(
        ["overall_match", "source_match_id", "team_key", "player_id"], kind="stable"
    ).reset_index(drop=True)


def _player_features(player_history: list[float], team_history: list[float]) -> dict[str, float]:
    values = np.asarray(player_history, dtype=float)
    return {
        "matches_before": float(len(values)),
        "prev_avg_finishes": float(values.mean()) if len(values) else 0.0,
        "recent3_avg_finishes": float(values[-3:].mean()) if len(values) else 0.0,
        "prev_std_finishes": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
        "team_prev_avg_finishes": float(np.mean(team_history)) if team_history else 0.0,
    }


def build_player_training_frame(player_matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Create one strictly pre-match feature row per player-match record."""
    rows = _matched_player_rows(player_matches, fixtures)
    if rows.empty:
        return pd.DataFrame(columns=[*PLAYER_FEATURES, "finishes", "overall_match", "source_match_id", "player_id", "player", "team"])
    player_history: dict[str, list[float]] = {}
    team_history: dict[str, list[float]] = {}
    output: list[dict[str, Any]] = []
    for (order, match_id), match in rows.groupby(["overall_match", "source_match_id"], sort=True):
        current_team_totals: dict[str, float] = {}
        current_player_values: list[tuple[str, float]] = []
        for _, row in match.iterrows():
            player_key = str(row["player_id"])
            team_key = str(row["team_key"])
            state = _player_features(player_history.get(player_key, []), team_history.get(team_key, []))
            output.append({
                **state,
                "finishes": float(row["finishes"]),
                "overall_match": float(order),
                "source_match_id": str(match_id),
                "player_id": player_key,
                "player": str(row["player"]),
                "team": str(row["team"]),
                "team_key": team_key,
            })
            current_player_values.append((player_key, float(row["finishes"])))
            current_team_totals[team_key] = current_team_totals.get(team_key, 0.0) + float(row["finishes"])
        # All players in a match use the same pre-match snapshot. Update only
        # after every row for the match has been featurized.
        for player_key, finishes in current_player_values:
            player_history.setdefault(player_key, []).append(finishes)
        for team_key, total_finishes in current_team_totals.items():
            team_history.setdefault(team_key, []).append(total_finishes)
    return pd.DataFrame(output)


def build_player_prediction_frame(
    player_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: list[str],
    before_overall_match: float | None,
) -> pd.DataFrame:
    """Build current player features from BMSD matches before the forecast."""
    rows = _matched_player_rows(player_matches, fixtures)
    if rows.empty:
        return pd.DataFrame(columns=["player_id", "player", "team", *PLAYER_FEATURES])
    if before_overall_match is not None and np.isfinite(before_overall_match):
        rows = rows[rows["overall_match"].lt(before_overall_match)].copy()
    team_keys = {canonical_team_key(team) for team in teams}
    rows = rows[rows["team_key"].isin(team_keys)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["player_id", "player", "team", *PLAYER_FEATURES])
    team_history = rows.groupby(["team_key", "source_match_id", "overall_match"], as_index=False)["finishes"].sum()
    team_history_by_key = {
        team_key: subset.sort_values("overall_match", kind="stable")["finishes"].astype(float).tolist()
        for team_key, subset in team_history.groupby("team_key", sort=False)
    }
    outputs = []
    for (player_id, team_key), subset in rows.groupby(["player_id", "team_key"], sort=False):
        subset = subset.sort_values("overall_match", kind="stable")
        player_values = subset["finishes"].astype(float).tolist()
        state = _player_features(player_values, team_history_by_key.get(team_key, []))
        outputs.append({
            "player_id": str(player_id),
            "player": str(subset.iloc[-1]["player"]),
            "team": str(subset.iloc[-1]["team"]),
            **state,
        })
    return pd.DataFrame(outputs)


def train_player_finish_model(
    player_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    features_path: str | Path = FEATURES_PATH,
    metrics_path: str | Path = METRICS_PATH,
    metadata_path: str | Path = METADATA_PATH,
    holdout_matches: int = 6,
) -> dict[str, Any]:
    """Compare a rolling-average baseline with a small random forest."""
    frame = build_player_training_frame(player_matches, fixtures)
    matches = sorted(frame["overall_match"].dropna().unique().tolist())
    if len(matches) <= holdout_matches + 6 or frame.empty:
        raise ValueError("Not enough completed player matches to train a chronological finish model.")
    holdout = set(matches[-holdout_matches:])
    train = frame[~frame["overall_match"].isin(holdout)].copy()
    test = frame[frame["overall_match"].isin(holdout)].copy()
    X_train = train[PLAYER_FEATURES].astype(float)
    y_train = train["finishes"].astype(float)
    X_test = test[PLAYER_FEATURES].astype(float)
    y_test = test["finishes"].astype(float)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    model_prediction = np.clip(model.predict(X_test), 0.0, None)
    baseline_center = float(y_train.mean())
    baseline_prediction = test["prev_avg_finishes"].where(test["matches_before"].gt(0), baseline_center).to_numpy(dtype=float)

    def metrics(name: str, prediction: np.ndarray) -> dict[str, Any]:
        return {
            "model": name,
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "r2": float(r2_score(y_test, prediction)),
            "holdout_matches": holdout_matches,
            "holdout_players": int(len(test)),
        }

    comparison = pd.DataFrame([
        metrics("Rolling average baseline", baseline_prediction),
        metrics("Random Forest regressor", model_prediction),
    ]).sort_values(["mae", "rmse"], ascending=True).reset_index(drop=True)
    selected = str(comparison.iloc[0]["model"])
    # The saved forest is used only when it beats the no-complexity baseline;
    # otherwise the app uses the rolling-average estimate.
    model_path, features_path = Path(model_path), Path(features_path)
    metrics_path, metadata_path = Path(metrics_path), Path(metadata_path)
    for path in (model_path, features_path, metrics_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(frame[PLAYER_FEATURES].astype(float), frame["finishes"].astype(float))
    comparison.to_csv(metrics_path, index=False)
    joblib.dump(final_model, model_path)
    joblib.dump(PLAYER_FEATURES, features_path)
    metadata = {
        "selected_method": selected,
        "target": "player finishes in one match",
        "metric": "lowest chronological holdout MAE",
        "match_count": len(matches),
        "row_count": int(len(frame)),
        "holdout_match_orders": [float(value) for value in matches[-holdout_matches:]],
        "features": PLAYER_FEATURES,
        "mvp_award_probability": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"comparison": comparison, "selected_method": selected, "metadata": metadata, "model": final_model}
