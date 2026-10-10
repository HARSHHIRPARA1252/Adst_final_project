"""Chronological model comparison, calibration, evaluation, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .data_preprocessing import load_all_historical_data
    from .feature_engineering import FEATURE_COLS, build_pre_match_features
except ImportError:  # Running as ``python src/model_training.py``.
    from data_preprocessing import load_all_historical_data
    from feature_engineering import FEATURE_COLS, build_pre_match_features


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "model_results"
MODEL_PATH = MODEL_DIR / "bgmi_winning_probability_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessing_pipeline.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.joblib"
METRICS_PATH = RESULTS_DIR / "model_comparison.csv"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"


def make_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Build a small set of calibrated, probability-oriented classifiers."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=4,
            random_state=random_state,
        ),
    }


def temporal_group_folds(
    match_keys: pd.Series | list[str],
    n_splits: int,
    validation_matches: int = 6,
    minimum_training_matches: int = 12,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window folds that never split a match across folds."""
    keys = pd.Series(match_keys).astype(str).reset_index(drop=True)
    ordered = keys.drop_duplicates().tolist()
    if len(ordered) < minimum_training_matches + validation_matches:
        raise ValueError(
            f"Need at least {minimum_training_matches + validation_matches} complete matches "
            "for temporal calibration."
        )
    first_start = len(ordered) - n_splits * validation_matches
    if first_start < minimum_training_matches:
        raise ValueError("Not enough complete matches for the requested temporal calibration folds.")
    folds = []
    for fold in range(n_splits):
        start = first_start + fold * validation_matches
        end = start + validation_matches
        train_match_keys = set(ordered[:start])
        validation_match_keys = set(ordered[start:end])
        train_idx = np.flatnonzero(keys.isin(train_match_keys).to_numpy())
        validation_idx = np.flatnonzero(keys.isin(validation_match_keys).to_numpy())
        folds.append((train_idx, validation_idx))
    return folds


def _field_probabilities(probabilities: np.ndarray, match_keys: pd.Series) -> np.ndarray:
    """Condition calibrated team scores to a one-winner probability per lobby."""
    output = np.zeros_like(probabilities, dtype=float)
    groups = match_keys.reset_index(drop=True).astype(str)
    for key in groups.unique():
        idx = np.flatnonzero(groups.eq(key).to_numpy())
        values = np.clip(probabilities[idx], 1e-12, 1.0)
        total = values.sum()
        output[idx] = values / total if total > 0 else 1.0 / max(1, len(idx))
    return output


def _evaluate(name: str, estimator: Any, X_test: np.ndarray, y_test: np.ndarray, keys: pd.Series) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    raw = np.asarray(estimator.predict_proba(X_test))[:, 1]
    raw = np.clip(raw, 1e-8, 1 - 1e-8)
    conditional = _field_probabilities(raw, keys)

    predicted_winner = np.zeros(len(y_test), dtype=int)
    group_series = keys.reset_index(drop=True).astype(str)
    for key in group_series.unique():
        indexes = np.flatnonzero(group_series.eq(key).to_numpy())
        predicted_winner[indexes[int(np.argmax(conditional[indexes]))]] = 1

    true_winner_probabilities = conditional[np.asarray(y_test, dtype=int) == 1]
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predicted_winner)),
        "precision": float(precision_score(y_test, predicted_winner, zero_division=0)),
        "recall": float(recall_score(y_test, predicted_winner, zero_division=0)),
        "f1": float(f1_score(y_test, predicted_winner, zero_division=0)),
        "top1_accuracy": float(predicted_winner[np.asarray(y_test, dtype=int) == 1].mean()),
        "roc_auc": float(roc_auc_score(y_test, raw)) if len(np.unique(y_test)) == 2 else np.nan,
        "log_loss": float(log_loss(y_test, raw, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_test, raw)),
        "match_log_loss": float(-np.log(np.clip(true_winner_probabilities, 1e-12, 1)).mean()),
        "evaluation_matches": int(group_series.nunique()),
    }
    return metrics, raw, predicted_winner


def _mean_feature_importance(estimator: Any, feature_names: list[str]) -> pd.DataFrame:
    """Average model-native importance over the temporal calibration ensemble."""
    fitted_models = getattr(estimator, "calibrated_classifiers_", [])
    values: list[np.ndarray] = []
    for fitted in fitted_models:
        model = getattr(fitted, "estimator", None)
        if model is None:
            continue
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importance = np.abs(np.asarray(model.coef_, dtype=float)).mean(axis=0)
        else:
            continue
        if len(importance) == len(feature_names):
            values.append(importance)
    if not values:
        return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})
    mean_values = np.mean(values, axis=0)
    total = mean_values.sum()
    if total > 0:
        mean_values = mean_values / total
    return pd.DataFrame({"feature": feature_names, "importance": mean_values}).sort_values("importance", ascending=False)


def _write_reports(
    results: pd.DataFrame,
    chosen_name: str,
    raw_probabilities: np.ndarray,
    y_test: np.ndarray,
    predicted_winner: np.ndarray,
    features: pd.DataFrame,
    feature_names: list[str],
    match_keys: pd.Series,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "model_comparison.csv", index=False)

    chosen = results.loc[results["model"].eq(chosen_name)].iloc[0]
    selected_prob = _field_probabilities(raw_probabilities, match_keys)
    predictions = pd.DataFrame({
        "match_key": match_keys.astype(str).values,
        "y_true": y_test.astype(int),
        "predicted_win": predicted_winner.astype(int),
        "raw_probability": raw_probabilities,
        "match_probability": selected_prob,
    })
    if "team" in features.columns:
        predictions.insert(1, "team", features["team"].astype(str).values)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    matrix = confusion_matrix(y_test, predicted_winner, labels=[0, 1])
    pd.DataFrame(matrix, index=["actual_no_win", "actual_win"], columns=["predicted_no_win", "predicted_win"]).to_csv(output_dir / "confusion_matrix.csv")
    _mean_feature_importance = features.attrs.get("importance")
    if isinstance(_mean_feature_importance, pd.DataFrame):
        _mean_feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Model comparison: probability log loss is the primary selection metric.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ordered = results.sort_values("match_log_loss")
    axes[0].bar(ordered["model"], ordered["match_log_loss"], color="#9c7cff")
    axes[0].set_title("Held-out match log loss (lower is better)")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(ordered["model"], ordered["top1_accuracy"], color="#38c89a")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Top-one winner accuracy")
    axes[1].tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, raw_probabilities)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(fpr, tpr, color="#a78bfa", label=f"{chosen_name} (AUC {chosen['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve · chronological holdout")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    curve_true, curve_probability = calibration_curve(y_test, raw_probabilities, n_bins=5, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(curve_probability, curve_true, marker="o", color="#38c89a", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777", label="Perfect calibration")
    ax.set(xlabel="Mean predicted win probability", ylabel="Observed win frequency", title="Reliability diagram")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    importance = features.attrs.get("importance")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        show = importance.sort_values("importance").tail(10)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.barh(show["feature"], show["importance"], color="#54b8ff")
        ax.set_title("Model feature importance (relative, not causal)")
        fig.tight_layout()
        fig.savefig(output_dir / "feature_importance.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def train_and_evaluate(
    data_dir: str | Path | None = None,
    model_dir: str | Path = MODEL_DIR,
    results_dir: str | Path = RESULTS_DIR,
    holdout_matches: int = 6,
) -> dict[str, Any]:
    """Compare models on later matches, calibrate temporally, and save artifacts.

    The final holdout is excluded from model comparison fits.  Within the
    development period, calibration windows expand forward by complete match
    groups, so no future match labels enter an earlier fold.
    """
    data_dir = Path(data_dir or PROJECT_ROOT / "data")
    model_dir = Path(model_dir)
    results_dir = Path(results_dir)
    history = load_all_historical_data(data_dir)
    if history.empty:
        raise ValueError("No usable match-level BGIS/BMPS data was found.")
    processed_path = data_dir / "processed" / "tournament_history.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(processed_path, index=False)

    features = build_pre_match_features(history)
    X_raw = features[FEATURE_COLS].astype(float)
    y = features["win"].astype(int).to_numpy()
    match_keys = features["match_key"].astype(str).reset_index(drop=True)
    ordered_keys = match_keys.drop_duplicates().tolist()
    if len(ordered_keys) <= holdout_matches + 12:
        raise ValueError("The history is too short for a 12-match training period and a later holdout.")
    test_keys = set(ordered_keys[-holdout_matches:])
    test_mask = match_keys.isin(test_keys).to_numpy()
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    train_keys = match_keys.iloc[train_idx].reset_index(drop=True)

    preprocessor_for_eval = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    X_train = preprocessor_for_eval.fit_transform(X_raw.iloc[train_idx])
    X_test = preprocessor_for_eval.transform(X_raw.iloc[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]
    fold_list = temporal_group_folds(train_keys, n_splits=3, validation_matches=6, minimum_training_matches=12)

    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    winners: dict[str, np.ndarray] = {}
    for name, estimator in make_candidate_models().items():
        calibrated = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=fold_list,
            ensemble=True,
        )
        calibrated.fit(X_train, y_train)
        metrics, raw_probability, predicted_winner = _evaluate(
            name, calibrated, X_test, y_test, match_keys.iloc[test_idx].reset_index(drop=True)
        )
        results.append(metrics)
        fitted[name] = calibrated
        probabilities[name] = raw_probability
        winners[name] = predicted_winner

    comparison = pd.DataFrame(results).sort_values(["match_log_loss", "brier_score"], ascending=True).reset_index(drop=True)
    chosen_name = str(comparison.iloc[0]["model"])

    # Refit the selected model with forward-only temporal calibration across
    # all available history. BMSD rows enter here only after they are imported.
    final_folds = temporal_group_folds(
        match_keys, n_splits=4, validation_matches=6, minimum_training_matches=12
    )
    final_preprocessor = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    X_all = final_preprocessor.fit_transform(X_raw)
    final_model = CalibratedClassifierCV(
        estimator=make_candidate_models()[chosen_name],
        method="sigmoid",
        cv=final_folds,
        ensemble=True,
    )
    final_model.fit(X_all, y)

    selected_raw = probabilities[chosen_name]
    selected_winners = winners[chosen_name]
    importance = _mean_feature_importance(final_model, FEATURE_COLS)
    report_features = features.iloc[test_idx].copy()
    report_features.attrs["importance"] = importance
    _write_reports(
        comparison,
        chosen_name,
        selected_raw,
        y_test,
        selected_winners,
        report_features,
        FEATURE_COLS,
        match_keys.iloc[test_idx].reset_index(drop=True),
        results_dir,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / "bgmi_winning_probability_model.joblib")
    joblib.dump(final_preprocessor, model_dir / "preprocessing_pipeline.joblib")
    joblib.dump(FEATURE_COLS, model_dir / "feature_names.joblib")
    metadata = {
        "selected_model": chosen_name,
        "target": "win = 1 for the match-winning team (WWCD); one winner per completed match",
        "calibration": "sigmoid calibration on expanding chronological match-group folds",
        "selection_metric": "lowest held-out conditional match log loss, Brier score tie-breaker",
        "random_state": RANDOM_STATE,
        "historical_matches": int(len(ordered_keys)),
        "historical_rows": int(len(features)),
        "test_matches": int(match_keys.iloc[test_idx].nunique()),
        "test_rows": int(len(test_idx)),
        "test_target_prevalence": float(y_test.mean()),
        "feature_names": FEATURE_COLS,
        "test_first_match": str(match_keys.iloc[test_idx].iloc[0]),
        "test_last_match": str(match_keys.iloc[test_idx].iloc[-1]),
    }
    with (results_dir / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return {
        "comparison": comparison,
        "selected_model": chosen_name,
        "final_model": final_model,
        "preprocessor": final_preprocessor,
        "feature_names": FEATURE_COLS,
        "feature_importance": importance,
        "metadata": metadata,
        "test_predictions": pd.read_csv(results_dir / "test_predictions.csv"),
    }


def main() -> None:
    result = train_and_evaluate()
    print("Chronological model comparison (final six matches held out):")
    print(result["comparison"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Selected: {result['selected_model']}")
    print(f"Model: {MODEL_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")
    print(f"Reports: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

