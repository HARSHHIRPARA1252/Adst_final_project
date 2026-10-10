"""Prediction helpers shared by the Flask app and notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from .feature_engineering import FEATURE_COLS, feature_snapshot_for_team
except ImportError:
    from feature_engineering import FEATURE_COLS, feature_snapshot_for_team


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "bgmi_winning_probability_model.joblib"
DEFAULT_PREPROCESSOR_PATH = Path(__file__).resolve().parents[1] / "models" / "preprocessing_pipeline.joblib"
DEFAULT_FEATURES_PATH = Path(__file__).resolve().parents[1] / "models" / "feature_names.joblib"


def load_predictor(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    preprocessor_path: str | Path = DEFAULT_PREPROCESSOR_PATH,
    feature_path: str | Path = DEFAULT_FEATURES_PATH,
) -> tuple[Any, Any, list[str]]:
    """Load trained model, preprocessing pipeline, and ordered input fields."""
    for path in (model_path, preprocessor_path, feature_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Required prediction artifact is missing: {path}")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_names = list(joblib.load(feature_path))
    return model, preprocessor, feature_names


def _input_frame(feature_values: dict[str, object], feature_names: list[str]) -> pd.DataFrame:
    row: dict[str, float] = {}
    for name in feature_names:
        value = feature_values.get(name, 0.0)
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric) or not np.isfinite(float(numeric)):
            raise ValueError(f"Feature '{name}' must be a finite number.")
        row[name] = float(numeric)
    return pd.DataFrame([row], columns=feature_names)


def predict_single_team(
    model: Any,
    feature_dict: dict[str, object],
    feature_names: list[str] | None = None,
    preprocessor: Any | None = None,
) -> float:
    """Return calibrated probability that one team wins its next match."""
    names = list(feature_names or FEATURE_COLS)
    values = _input_frame(feature_dict, names)
    matrix = preprocessor.transform(values) if preprocessor is not None else values
    probabilities = np.asarray(model.predict_proba(matrix))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Loaded classifier does not provide binary class probabilities.")
    return float(np.clip(probabilities[0, 1], 0.0, 1.0))


def predict_team_from_history(
    model: Any,
    preprocessor: Any,
    history: pd.DataFrame,
    team: str,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Create a leakage-safe next-match estimate from completed team records."""
    names = list(feature_names or FEATURE_COLS)
    features = feature_snapshot_for_team(history, team)
    probability = predict_single_team(model, features, names, preprocessor)
    return {"team": team, "winning_probability": probability, "features": features}


def predict_match_probabilities(
    model: Any,
    feature_df: pd.DataFrame,
    feature_names: list[str],
    preprocessor: Any | None = None,
    match_col: str = "match_key",
    team_col: str = "team",
) -> pd.DataFrame:
    """Estimate and condition each complete match lobby to one winner.

    The classifier produces calibrated individual team probabilities.  For a
    supplied complete lobby, these are normalized so the joint match outcome
    sums to one.  Do not use this normalization for a lone-team request.
    """
    frame = feature_df.copy()
    matrix = frame.reindex(columns=feature_names).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if preprocessor is not None:
        matrix = preprocessor.transform(matrix)
    frame["raw_probability"] = model.predict_proba(matrix)[:, 1]
    frame["winning_probability"] = frame.groupby(match_col, group_keys=False)["raw_probability"].transform(
        lambda values: values / values.sum() if values.sum() > 0 else np.repeat(1 / len(values), len(values))
    )
    return frame[[team_col, match_col, "raw_probability", "winning_probability"]].reset_index(drop=True)


def get_feature_importance(model: Any, feature_names: list[str] | None = None) -> list[dict[str, float | str]]:
    """Return model-native global importance, averaged across calibrators.

    This is a model-level explanation of which inputs the estimator uses most;
    it is not a causal explanation of an individual team's result.
    """
    names = list(feature_names or FEATURE_COLS)
    calibrated = getattr(model, "calibrated_classifiers_", [])
    importance_vectors = []
    for fitted in calibrated:
        estimator = getattr(fitted, "estimator", None)
        if isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]
        if estimator is None:
            continue
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_, dtype=float)).mean(axis=0)
        else:
            continue
        if len(values) == len(names):
            importance_vectors.append(values)
    if not importance_vectors:
        return []
    values = np.mean(importance_vectors, axis=0)
    total = values.sum()
    if total:
        values = values / total
    order = np.argsort(values)[::-1]
    return [{"feature": names[i], "importance": float(values[i])} for i in order]


def get_top_contributing_features(
    model: Any,
    feature_dict: dict[str, object],
    feature_names: list[str] | None = None,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Compatibility helper returning global model importance in rank order."""
    items = get_feature_importance(model, feature_names)
    return [(str(item["feature"]), float(item["importance"])) for item in items[:top_n]]


def probability_category(prob_pct: float) -> str:
    """Legacy compatibility; the current UI avoids arbitrary probability bands."""
    return "Model estimate"

