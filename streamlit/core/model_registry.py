from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from core.live_features import compute_latest_features


# ============================================================
# PROJECT PATHS AND DL IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the exact architecture and inference functions
# used by src/dl.py. Importing dl.py does NOT train models.
from dl import make_model, standardize, score


# ============================================================
# MODEL INFORMATION
# ============================================================

@dataclass
class ModelInfo:
    id: str
    name: str
    kind: str = "classical_ml"
    horizon_min: int | None = None

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
    # Register model choices
    # --------------------------------------------------------

    def _register_models(self):

        models = [
            ("logistic_regression", "Logistic Regression"),
            ("random_forest", "Random Forest"),
            ("extra_trees", "Extra Trees"),
            ("gradient_boosting", "Gradient Boosting"),
            ("xgboost", "XGBoost"),
        ]

        for model_id, name in models:
            self._models[model_id] = ModelInfo(
                id=model_id,
                name=name,
                kind="classical_ml",
            )

        # Register the two DL architectures for every horizon.
        for horizon in (20, 30, 40, 50):

            for model_id, name in [
                ("tabular_resnet", "Tabular ResNet"),
                ("ft_transformer", "FT-Transformer"),
            ]:

                unique_id = f"{model_id}_{horizon}min"

                self._models[unique_id] = ModelInfo(
                    id=unique_id,
                    name=name,
                    kind="deep_learning",
                    horizon_min=horizon,
                )

    # --------------------------------------------------------
    # List models
    # --------------------------------------------------------

    def list_models(self):

        return [
            {
                "id": model.id,
                "name": model.name,
                "kind": model.kind,
                "horizon_min": model.horizon_min,
                "metrics": model.metrics,
            }
            for model in self._models.values()
        ]

    # --------------------------------------------------------
    # Load one traditional ML model
    # --------------------------------------------------------

    def load_real_model_from_joblib(
        self,
        model_id: str,
        joblib_path: str,
        name: str | None = None,
        horizon_min: int | None = None,
    ):

        print(f"Loading model: {joblib_path}")

        artifact = joblib.load(joblib_path)

        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        threshold = artifact.get("threshold", 0.5)

        def real_predict(window: list[dict]) -> float:

            result = compute_latest_features(window)

            if result is None or not result["is_ready"]:
                return 0.0

            features = result["features"]

            row = [
                features.get(column, 0.0)
                for column in feature_columns
            ]

            X = pd.DataFrame(
                [row],
                columns=feature_columns,
            )

            probability = model.predict_proba(X)[0, 1]

            return float(probability)

        if model_id not in self._models:
            self._models[model_id] = ModelInfo(
                id=model_id,
                name=name or model_id,
            )

        info = self._models[model_id]
        info.predict_fn = real_predict
        info.horizon_min = horizon_min
        info.metrics["threshold"] = threshold

        print(
            f"Loaded {model_id} "
            f"(threshold={threshold:.4f})"
        )

    # --------------------------------------------------------
    # Load all traditional ML models for one horizon
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
                print(f"WARNING: model not found: {path}")
                continue

            self.load_real_model_from_joblib(
                model_id=model_id,
                joblib_path=str(path),
                name=display_name,
                horizon_min=horizon_min,
            )

    # --------------------------------------------------------
    # Load one DL checkpoint
    # --------------------------------------------------------

    def load_dl_model(
        self,
        models_dir: str,
        model_id: str,
        horizon_min: int,
    ):

        models_dir = Path(models_dir)

        filename = (
            f"horizon_label_{horizon_min}min__"
            f"{model_id}.pt"
        )

        path = models_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"DL checkpoint not found: {path}"
            )

        print(f"Loading DL model: {path}")

        # The checkpoints were created by our own training
        # script and contain tensors and basic metadata.
        artifact = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        # Verify the checkpoint matches the selected horizon.
        if artifact["horizon_min"] != horizon_min:
            raise ValueError(
                "Checkpoint horizon does not match "
                "the selected prediction horizon."
            )

        if artifact["model_name"] != model_id:
            raise ValueError(
                "Checkpoint architecture does not match "
                "the selected model."
            )

        feature_columns = artifact["feature_columns"]

        # Reconstruct the exact trained architecture.
        model = make_model(
            artifact["model_name"],
            len(feature_columns),
            artifact["model_config"],
        )

        model.load_state_dict(artifact["state_dict"])
        model.eval()

        scaler_mean = artifact["scaler_mean"].numpy()
        scaler_scale = artifact["scaler_scale"].numpy()

        threshold = float(artifact["threshold"])
        batch_size = int(artifact["batch_size"])

        # ----------------------------------------------------
        # Real DL inference
        # ----------------------------------------------------

        def real_predict(window: list[dict]) -> float:

            result = compute_latest_features(window)

            if result is None or not result["is_ready"]:
                return 0.0

            features = result["features"]

            # Require precisely the saved training features.
            missing = [
                col for col in feature_columns
                if col not in features
            ]

            extra = [
                col for col in features
                if col not in feature_columns
            ]

            if missing or extra:
                raise ValueError(
                    "Feature mismatch for DL model. "
                    f"Missing: {missing}; "
                    f"unexpected: {extra}"
                )

            feature_frame = pd.DataFrame(
                [[features[col] for col in feature_columns]],
                columns=feature_columns,
            )

            X = standardize(
                feature_frame.to_numpy(),
                scaler_mean,
                scaler_scale,
            )

            X_tensor = torch.from_numpy(X)

            with torch.inference_mode():
                probabilities = score(
                    model,
                    X_tensor,
                    batch_size,
                )

            return float(probabilities[0])

        # Each horizon/architecture has its own registry ID.
        unique_id = f"{model_id}_{horizon_min}min"

        info = self._models[unique_id]

        info.predict_fn = real_predict
        info.metrics["threshold"] = threshold

        # Preserve actual test metrics from the checkpoint.
        test_metrics = artifact.get("test_metrics", {})

        for key in ("f1", "precision", "recall", "auc_pr"):
            if key in test_metrics:
                info.metrics[key] = test_metrics[key]

        print(
            f"Loaded {unique_id} "
            f"(threshold={threshold:.4f})"
        )

    # --------------------------------------------------------
    # Load both DL architectures for one horizon
    # --------------------------------------------------------

    def load_all_dl_models(
        self,
        models_dir: str,
        horizon_min: int,
    ):

        for model_id in (
            "tabular_resnet",
            "ft_transformer",
        ):

            self.load_dl_model(
                models_dir=models_dir,
                model_id=model_id,
                horizon_min=horizon_min,
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

        if model.predict_fn is None:
            raise RuntimeError(
                f"Real model '{model_id}' has not been loaded."
            )

        return model.predict_fn(window)


# ============================================================
# GLOBAL REGISTRY
# ============================================================

registry = ModelRegistry()