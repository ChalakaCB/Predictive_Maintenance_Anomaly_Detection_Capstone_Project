"""Train five classifiers for each of the four rebalanced MetroPT-3 targets.

Run: python src/traditional_ml.py
Inputs: data/processed/horizon_label_<N>min/{train,val,test}.csv
Outputs: docs/results/*.csv and src/models/*.joblib

Parameters and thresholds are selected on validation F1. The selected model
is saved without refitting on validation data. Metrics describe the existing
full-pool-SMOTE/random-split protocol, not unseen-event forecasting performance.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "docs" / "results"
MODELS_DIR = ROOT / "src" / "models"
TARGETS = tuple(f"horizon_label_{h}min" for h in (20, 30, 40, 50))
RANDOM_STATE = 42
N_JOBS = 4
DATA_PROTOCOL = "full_pool_smote_undersampling_random_split"
LABEL_DEFINITION = "failure_starts_within_horizon; in_failure_rows_are_negative"

# Small initial search: 22 parameter trials per target, 88 fits in total.
PARAMETER_GRIDS = {
    "lr": {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    "rf": {"max_depth": [None, 12], "min_samples_leaf": [1, 3]},
    "extra_trees": {"max_depth": [None, 12], "min_samples_leaf": [1, 3]},
    "gradient_boosting": {
        "learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31],
    },
    "xgboost": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
}


def read_split(target, split, feature_columns=None):
    """Read only the selected target's rebalanced CSV and keep column order."""
    frame = pd.read_csv(DATA_DIR / target / f"{split}.csv")
    excluded = {"timestamp", "failure_id", "in_failure_window", "split"}
    columns = [
        c for c in frame.columns
        if c not in excluded
        and not c.startswith(("horizon_label_", "history_ready_", "Unnamed:"))
    ]
    if feature_columns is not None and columns != feature_columns:
        raise ValueError(f"Feature columns differ in {target}/{split}.csv")
    X = frame[columns].astype(np.float32)
    y = frame[target]
    if set(y.unique()) != {0, 1} or not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"Expected finite features and both binary classes: {target}/{split}")
    return X, y.astype(np.int8)


def make_model(name, params):
    """Keep scaling inside the LR pipeline so it is fitted on train only."""
    if name == "lr":
        estimator = LogisticRegression(
            max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE, **params
        )
    elif name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=200, n_jobs=N_JOBS, random_state=RANDOM_STATE, **params
        )
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=200, n_jobs=N_JOBS, random_state=RANDOM_STATE, **params
        )
    elif name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=200, early_stopping=False, random_state=RANDOM_STATE, **params
        )
    elif name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=200, subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=N_JOBS, random_state=RANDOM_STATE, **params
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    steps = [("scaler", StandardScaler())] if name == "lr" else []
    return Pipeline([*steps, ("model", estimator)])


def choose_threshold(y, scores):
    """Maximise validation positive-class F1; ties prefer closest to 0.5."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1], denominator,
        out=np.zeros_like(denominator), where=denominator > 0,
    )
    candidates = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    best = candidates[np.argmin(np.abs(thresholds[candidates] - 0.5))]
    return float(thresholds[best])


def evaluate(y, scores, threshold):
    """All precision/recall/F1 values refer to the positive class (label 1)."""
    predicted = (scores >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparisons, trials = [], []
    versions = {
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }

    for target in TARGETS:
        X_train, y_train = read_split(target, "train")
        feature_columns = X_train.columns.tolist()
        X_val, y_val = read_split(target, "val", feature_columns)
        print(f"\n{target}: train={len(y_train):,}, val={len(y_val):,}, features={len(feature_columns)}", flush=True)

        for name, grid in PARAMETER_GRIDS.items():
            best = None
            first_trial = len(trials)
            print(f"  {name}: tuning {len(ParameterGrid(grid))} configurations", flush=True)
            for trial_id, params in enumerate(ParameterGrid(grid), start=1):
                model = make_model(name, params)
                model.fit(X_train, y_train)
                val_scores = model.predict_proba(X_val)[:, 1]
                threshold = choose_threshold(y_val, val_scores)
                metrics = evaluate(y_val, val_scores, threshold)
                trials.append({
                    "target": target, "model": name, "trial_id": trial_id,
                    "params": json.dumps(params, sort_keys=True),
                    "threshold": threshold, "selected": False,
                    **{f"val_{k}": v for k, v in metrics.items()},
                })
                # Test data never participate in parameter or threshold selection.
                rank = (metrics["f1"], metrics["precision"])
                if best is None or rank > best["rank"]:
                    best = {
                        "model": model, "params": params, "threshold": threshold,
                        "metrics": metrics, "rank": rank, "trial_id": trial_id,
                    }

            trials[first_trial + best["trial_id"] - 1]["selected"] = True
            X_test, y_test = read_split(target, "test", feature_columns)
            test_scores = best["model"].predict_proba(X_test)[:, 1]
            test_metrics = evaluate(y_test, test_scores, best["threshold"])

            model_path = MODELS_DIR / f"{target}__{name}.joblib"
            artifact = {
                "model": best["model"], "target": target, "model_name": name,
                "feature_columns": feature_columns, "best_params": best["params"],
                "estimator_params": best["model"].named_steps["model"].get_params(),
                "threshold": best["threshold"], "score_type": "probability",
                "selection_metric": "validation_positive_f1_then_precision",
                "fit_split": "train", "threshold_source": "val",
                "data_protocol": DATA_PROTOCOL, "label_definition": LABEL_DEFINITION,
                "random_state": RANDOM_STATE, "versions": versions,
                "val_metrics": best["metrics"], "test_metrics": test_metrics,
            }
            joblib.dump(artifact, model_path, compress=3)
            comparisons.append({
                "target": target, "model": name,
                "best_params": json.dumps(best["params"], sort_keys=True),
                "threshold": best["threshold"], **test_metrics,
            })
            # Write after each combination so completed work is visible during a run.
            pd.DataFrame(comparisons).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
            pd.DataFrame(trials).to_csv(RESULTS_DIR / "tuning_trials.csv", index=False)
            print(
                f"    test precision={test_metrics['precision']:.4f}, "
                f"recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}; "
                f"threshold={best['threshold']:.4f}; saved {model_path.name}", flush=True,
            )

    print(f"\nFinished: {len(comparisons)} models, {len(trials)} parameter trials.", flush=True)
    print(f"Results: {RESULTS_DIR}\nModels: {MODELS_DIR}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=N_JOBS):
        main()
