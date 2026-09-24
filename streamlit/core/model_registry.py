from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import pandas as pd

from core.live_features import compute_latest_features


# ============================================================
# MODEL INFORMATION
# ============================================================

@dataclass
class ModelInfo:
    id: str
    name: str
    kind: str = "classical_ml"

    metrics: dict = field(
        default_factory=lambda: {
            "f1": None,
            "precision": None,
            "recall": None,
            "auc_pr": None,
            "threshold": None,
        }
    )

    predict_fn: Optional[Callable] = None


# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry:

    def __init__(self):
        self._models: dict[str, ModelInfo] = {}

        self._register_models()


    # --------------------------------------------------------
    # Register the five trained algorithms
    # --------------------------------------------------------

    def _register_models(self):

        models = [
            (
                "logistic_regression",
                "Logistic Regression",
            ),
            (
                "random_forest",
                "Random Forest",
            ),
            (
                "extra_trees",
                "Extra Trees",
            ),
            (
                "gradient_boosting",
                "Gradient Boosting",
            ),
            (
                "xgboost",
                "XGBoost",
            ),
        ]

        for model_id, name in models:

            self._models[model_id] = ModelInfo(
                id=model_id,
                name=name,
                kind="classical_ml",
            )


    # --------------------------------------------------------
    # List models
    # --------------------------------------------------------

    def list_models(self) -> list[dict]:

        return [
            {
                "id": model.id,
                "name": model.name,
                "kind": model.kind,
                "metrics": model.metrics,
            }
            for model in self._models.values()
        ]


    # --------------------------------------------------------
    # Load ONE real .joblib model
    # --------------------------------------------------------

    def load_real_model_from_joblib(
        self,
        model_id: str,
        joblib_path: str,
        name: str | None = None,
    ):

        print(f"Loading model: {joblib_path}")

        artifact = joblib.load(joblib_path)

        # Your traditional_ml.py saved these.
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]

        # Threshold selected during model training.
        threshold = artifact.get(
            "threshold",
            0.5
        )

        # ----------------------------------------------------
        # This function is called whenever Streamlit asks
        # the model for a prediction.
        # ----------------------------------------------------

        def real_predict(window: list[dict]) -> float:

            # Raw sensor rows
            #       ↓
            # 173 engineered features
            result = compute_latest_features(window)

            # Not enough history yet.
            if result is None:
                return 0.0

            if not result["is_ready"]:
                return 0.0

            features = result["features"]

            # ------------------------------------------------
            # VERY IMPORTANT:
            #
            # Use the exact feature order saved with the model.
            # ------------------------------------------------

            row = [
                features.get(column, 0.0)
                for column in feature_columns
            ]

            X = pd.DataFrame(
                [row],
                columns=feature_columns,
            )

            # ------------------------------------------------
            # REAL MODEL PREDICTION
            # ------------------------------------------------

            probability = model.predict_proba(X)[0, 1]

            return float(probability)


        # Make sure model exists in registry.
        if model_id not in self._models:

            self._models[model_id] = ModelInfo(
                id=model_id,
                name=name or model_id,
            )

        # Connect the real prediction function.
        self._models[model_id].predict_fn = real_predict

        # Save the training threshold.
        self._models[model_id].metrics["threshold"] = threshold

        print(
            f"Loaded {model_id} "
            f"(threshold={threshold:.4f})"
        )


    # --------------------------------------------------------
    # Load all five models for ONE horizon
    # --------------------------------------------------------

    def load_all_real_models(
        self,
        models_dir: str,
        horizon_min: int,
    ):

        models_dir = Path(models_dir)

        model_files = {
            "lr": (
                "logistic_regression",
                "Logistic Regression",
            ),
            "rf": (
                "random_forest",
                "Random Forest",
            ),
            "extra_trees": (
                "extra_trees",
                "Extra Trees",
            ),
            "gradient_boosting": (
                "gradient_boosting",
                "Gradient Boosting",
            ),
            "xgboost": (
                "xgboost",
                "XGBoost",
            ),
        }

        for file_id, (model_id, display_name) in model_files.items():

            filename = (
                f"horizon_label_{horizon_min}min__"
                f"{file_id}.joblib"
            )

            path = models_dir / filename

            if not path.exists():

                print(
                    f"WARNING: model not found: {path}"
                )

                continue

            self.load_real_model_from_joblib(
                model_id=model_id,
                joblib_path=str(path),
                name=display_name,
            )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict(
        self,
        model_id: str,
        window: list[dict],
    ) -> float:

        if model_id not in self._models:

            raise ValueError(
                f"Unknown model: {model_id}"
            )

        model = self._models[model_id]

        # Real model loaded?
        if model.predict_fn is not None:

            return model.predict_fn(window)

        raise RuntimeError(
            f"Real model '{model_id}' has not been loaded."
        )


# ============================================================
# GLOBAL REGISTRY
# ============================================================

registry = ModelRegistry()