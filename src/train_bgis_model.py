"""
train_bgis_model.py
====================
BGMI Win Probability - BGIS Tournament Model Training Script
-------------------------------------------------------------
Loads bgis_features.csv, trains 3 classifiers (Logistic Regression,
Random Forest, Gradient Boosting), selects best by ROC-AUC, calibrates
it with isotonic regression, and saves the final artefacts.

Usage:
    cd d:/BGI_Win_Probability
    python src/train_bgis_model.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 on Windows consoles that default to cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# -- Make src/ importable ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import joblib
import numpy  as np
import pandas as pd

from sklearn.calibration     import CalibratedClassifierCV
from sklearn.ensemble        import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

# =============================================================================
# 1.  PATHS
# =============================================================================
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES_CSV  = os.path.join(DATA_DIR,   "bgis_features.csv")
MODEL_SAVE    = os.path.join(MODELS_DIR, "bgmi_winning_probability_model.joblib")
FEAT_SAVE     = os.path.join(MODELS_DIR, "bgis_feature_names.joblib")
METRICS_SAVE  = os.path.join(DATA_DIR,   "model_comparison.csv")

# =============================================================================
# 2.  FEATURE LIST
# =============================================================================
FEATURE_COLS = [
    "prev_avg_points",
    "prev_avg_kills",
    "prev_win_rate",
    "recent3_avg_points",
    "recent3_avg_kills",
    "recent3_win_rate",
    "relative_points",
    "relative_kills",
    "relative_recent_points",
    "relative_recent_kills",
]
TARGET_COL   = "win"
SORT_COL     = "match_num"
TRAIN_FRAC   = 0.65
RANDOM_STATE = 42

# =============================================================================
# 3.  LOAD & VALIDATE DATA
# =============================================================================
SEP = "=" * 65
DIV = "-" * 65

print(SEP)
print("  BGIS Win Probability - Model Training")
print(SEP)

df = pd.read_csv(FEATURES_CSV)
print(f"\n[INFO] Loaded {len(df):,} rows from: {FEATURES_CSV}")
print(f"[INFO] Columns: {list(df.columns)}")

missing = [c for c in FEATURE_COLS + [TARGET_COL, SORT_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

pre_len = len(df)
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
print(f"[INFO] Dropped {pre_len - len(df)} rows with NaN values -> {len(df):,} usable rows")

n_wins  = int(df[TARGET_COL].sum())
n_total = len(df)
win_pct = 100.0 * n_wins / n_total
print(f"\n[INFO] Target distribution -> win=1: {n_wins:,} ({win_pct:.2f}%)  |  win=0: {n_total-n_wins:,} ({100-win_pct:.2f}%)")

# =============================================================================
# 4.  CHRONOLOGICAL TRAIN / TEST SPLIT  (65% train, 35% test)
# =============================================================================
df_sorted = df.sort_values(SORT_COL).reset_index(drop=True)
split_idx = int(len(df_sorted) * TRAIN_FRAC)

train_df  = df_sorted.iloc[:split_idx]
test_df   = df_sorted.iloc[split_idx:]

X_train = train_df[FEATURE_COLS].values
y_train = train_df[TARGET_COL].values.astype(int)
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET_COL].values.astype(int)

print(f"\n[INFO] Chronological split (sort key: '{SORT_COL}')")
print(f"       Train : {len(X_train):,} rows  (match_num range: {train_df[SORT_COL].min()} - {train_df[SORT_COL].max()})")
print(f"       Test  : {len(X_test):,} rows  (match_num range: {test_df[SORT_COL].min()} - {test_df[SORT_COL].max()})")
print(f"       Train wins: {y_train.sum()} ({100*y_train.mean():.2f}%)  |  Test wins: {y_test.sum()} ({100*y_test.mean():.2f}%)")

# =============================================================================
# 5.  MODEL DEFINITIONS
# =============================================================================
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=1,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    ),
}

# =============================================================================
# 6.  TRAIN & EVALUATE
# =============================================================================
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """Fit model on train set and evaluate on test set."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_te, y_pred),                    4),
        "precision": round(precision_score(y_te, y_pred, zero_division=0),  4),
        "recall":    round(recall_score(y_te, y_pred, zero_division=0),     4),
        "f1":        round(f1_score(y_te, y_pred, zero_division=0),         4),
        "roc_auc":   round(roc_auc_score(y_te, y_prob),                     4),
        "log_loss":  round(log_loss(y_te, y_prob),                          4),
        "brier":     round(brier_score_loss(y_te, y_prob),                  4),
    }
    return metrics, model


results       = []
fitted_models = {}

print("\n" + DIV)
print("  Training & Evaluating Models")
print(DIV)

for name, model in models.items():
    print(f"\n  > {name} ...", end="", flush=True)
    metrics, fitted = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    fitted_models[name] = fitted
    results.append(metrics)
    print(" done")
    print(f"      Accuracy : {metrics['accuracy']}")
    print(f"      Precision: {metrics['precision']}")
    print(f"      Recall   : {metrics['recall']}")
    print(f"      F1       : {metrics['f1']}")
    print(f"      ROC-AUC  : {metrics['roc_auc']}")
    print(f"      Log-Loss : {metrics['log_loss']}")
    print(f"      Brier    : {metrics['brier']}")

# =============================================================================
# 7.  SELECT BEST MODEL BY ROC-AUC
# =============================================================================
metrics_df = pd.DataFrame(results)
best_idx   = metrics_df["roc_auc"].idxmax()
best_name  = metrics_df.loc[best_idx, "model"]
best_model = fitted_models[best_name]

print("\n" + DIV)
print(f"  Best model by ROC-AUC: [*] {best_name}  (AUC={metrics_df.loc[best_idx,'roc_auc']})")
print(DIV)

# =============================================================================
# 8.  CALIBRATION  (isotonic, prefit)
# =============================================================================
print(f"\n[INFO] Calibrating '{best_name}' with CalibratedClassifierCV(isotonic, cv='prefit') ...")

calibrated = CalibratedClassifierCV(
    estimator=best_model,
    method="isotonic",
    cv="prefit",
)
calibrated.fit(X_test, y_test)   # calibrate on held-out test set

y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
y_pred_cal = calibrated.predict(X_test)
cal_metrics = {
    "model":     f"{best_name} (Calibrated)",
    "accuracy":  round(accuracy_score(y_test, y_pred_cal),                   4),
    "precision": round(precision_score(y_test, y_pred_cal, zero_division=0), 4),
    "recall":    round(recall_score(y_test, y_pred_cal, zero_division=0),    4),
    "f1":        round(f1_score(y_test, y_pred_cal, zero_division=0),        4),
    "roc_auc":   round(roc_auc_score(y_test, y_prob_cal),                    4),
    "log_loss":  round(log_loss(y_test, y_prob_cal),                         4),
    "brier":     round(brier_score_loss(y_test, y_prob_cal),                 4),
}
results.append(cal_metrics)
metrics_df = pd.DataFrame(results)

print(f"  Post-calibration ROC-AUC : {cal_metrics['roc_auc']}")
print(f"  Post-calibration Brier   : {cal_metrics['brier']}")
print(f"  Post-calibration Log-Loss: {cal_metrics['log_loss']}")

# =============================================================================
# 9.  SAVE ARTEFACTS
# =============================================================================
joblib.dump(calibrated,   MODEL_SAVE)
joblib.dump(FEATURE_COLS, FEAT_SAVE)
metrics_df.to_csv(METRICS_SAVE, index=False)

print("\n" + SEP)
print("  Saved artefacts")
print(f"  Model         -> {MODEL_SAVE}")
print(f"  Feature names -> {FEAT_SAVE}")
print(f"  Metrics CSV   -> {METRICS_SAVE}")
print(SEP)

# =============================================================================
# 10.  SUMMARY TABLE
# =============================================================================
print("\n  Full Metrics Comparison:")
print(metrics_df.to_string(index=False))
print("\n[DONE] BGIS model training complete.\n")
