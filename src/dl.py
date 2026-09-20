"""Train tabular ResNet and FT-Transformer: python src/dl.py.

Uses the same full-pool-rebalanced, random-split CSVs as traditional_ml.py.
This protocol does not measure generalisation to unseen failure events.
Import load_model/predict for deployment; importing this module never trains.
"""

import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rtdl_revisiting_models import FTTransformer, ResNet
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "docs" / "results"
MODELS_DIR = ROOT / "src" / "models"
TARGETS = tuple(f"horizon_label_{h}min" for h in (20, 30, 40, 50))
SEED = 42
WEIGHT_DECAY = 1e-4
MODEL_CONFIGS = {
    "tabular_resnet": {
        "n_blocks": 2, "d_block": 128, "d_hidden_multiplier": 2.0,
        "dropout1": 0.1, "dropout2": 0.0,
    },
    "ft_transformer": {
        "n_blocks": 1, "d_block": 32, "attention_n_heads": 4,
        "attention_dropout": 0.1, "ffn_d_hidden_multiplier": 4 / 3,
        "ffn_dropout": 0.1, "residual_dropout": 0.0,
    },
}
BATCH_SIZES = {"tabular_resnet": 2048, "ft_transformer": 512}
# Add only the six configurations supported by the previous validation search.
EXTRA_PARAMS = {
    "tabular_resnet": {
        20: {"n_blocks": 3, "d_block": 192, "learning_rate": 3e-4, "pos_weight": 25.0},
        30: {"learning_rate": 3e-4, "pos_weight": 25.0},
        40: {"learning_rate": 3e-4, "pos_weight": 25.0},
    },
    "ft_transformer": {
        20: {"n_blocks": 2, "learning_rate": 1e-3, "pos_weight": 1.0},
        30: {"d_block": 64, "learning_rate": 1e-3, "pos_weight": 25.0},
        40: {"d_block": 64, "learning_rate": 1e-3, "pos_weight": 1.0},
    },
}


def parameter_trials(name, horizon):
    """Four original trials per target, plus one extra for 20/30/40 minutes."""
    base = {
        **MODEL_CONFIGS[name], "batch_size": BATCH_SIZES[name],
        "weight_decay": WEIGHT_DECAY, "max_epochs": 12, "patience": 3,
        "lr_scheduler": False,
    }
    trials = [
        {**base, "learning_rate": lr, "pos_weight": weight}
        for lr in (1e-3, 3e-4) for weight in (1.0, 25.0)
    ]
    if horizon in EXTRA_PARAMS[name]:
        trials.append({
            **base, **EXTRA_PARAMS[name][horizon],
            "max_epochs": 40, "patience": 8, "lr_scheduler": True,
        })
    return trials


def make_model(name, n_features, config):
    """Use the authors' architectures; all 173 features are numerical inputs."""
    if name == "tabular_resnet":
        return ResNet(d_in=n_features, d_out=1, **config)
    if name == "ft_transformer":
        return FTTransformer(
            n_cont_features=n_features, cat_cardinalities=[], d_out=1, **config
        )
    raise ValueError(f"Unknown model: {name}")


def forward(model, X):
    logits = model(X, None) if isinstance(model, FTTransformer) else model(X)
    return logits.squeeze(-1)


@torch.inference_mode()
def score(model, X, batch_size=1024):
    """Return positive-class scores; no gradients or training-time dropout."""
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([
        forward(model, batch.to(device)).sigmoid().cpu()
        for batch in X.split(batch_size)
    ]).numpy()


def standardize(values, mean, scale):
    """The exact same float32 transform is used for training and deployment."""
    values = np.array(values, dtype=np.float32, copy=True)
    values -= mean
    values /= scale
    if not np.isfinite(values).all():
        raise ValueError("Model features must be finite after standardisation.")
    return values


def load_model(path, device="cpu"):
    """Load a trusted checkpoint once, then reuse it for successive windows."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(
        artifact["model_name"], len(artifact["feature_columns"]),
        artifact["model_config"],
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return {"model": model, "metadata": artifact}


def predict(predictor, feature_frame):
    """Predict one or more engineered windows, not raw sensor records.

    Column order is restored from the checkpoint. Missing/extra columns fail
    clearly so raw sensors, targets and audit metadata cannot enter unnoticed.
    """
    model, meta = predictor["model"], predictor["metadata"]
    columns = meta["feature_columns"]
    if not feature_frame.columns.is_unique or set(feature_frame.columns) != set(columns):
        raise ValueError("Provide exactly the saved feature columns, without labels or metadata.")
    if feature_frame.empty:
        raise ValueError("Provide at least one feature row.")
    X = standardize(
        feature_frame.loc[:, columns].to_numpy(),
        meta["scaler_mean"].numpy(), meta["scaler_scale"].numpy(),
    )
    scores = score(model, torch.from_numpy(X), meta["batch_size"])
    return pd.DataFrame({
        "risk_score": scores, "threshold": meta["threshold"],
        "alarm": scores >= meta["threshold"], "horizon_min": meta["horizon_min"],
    }, index=feature_frame.index)


def fit_trial(name, params, X_train, y_train, X_val, y_val, device):
    # Lazy imports keep deployment independent of the classical training code.
    try:
        from .traditional_ml import choose_threshold, evaluate
    except ImportError:
        from traditional_ml import choose_threshold, evaluate

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    config = {key: params[key] for key in MODEL_CONFIGS[name]}
    model = make_model(name, X_train.shape[1], config).to(device)
    parameters = model.make_parameter_groups() if isinstance(model, FTTransformer) else model.parameters()
    optimizer = torch.optim.AdamW(
        parameters, lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    ) if params["lr_scheduler"] else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params["pos_weight"], device=device))
    batch_size = params["batch_size"]
    best, stale = None, 0
    started = time.perf_counter()

    for epoch in range(1, params["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(y_train), device=device)
        batches = list(order.split(batch_size))
        # BatchNorm in ResNet needs >1 training row. Keep, rather than drop,
        # a final singleton by joining it to the preceding batch.
        if len(batches) > 1 and len(batches[-1]) == 1:
            last = batches.pop()
            batches[-1] = torch.cat([batches[-1], last])
        total_loss = torch.zeros((), device=device)
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(forward(model, X_train[indices]), y_train[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {name}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.detach() * len(indices)

        # Training can use faster CUDA matmul; checkpoint selection uses full
        # float32 precision to reduce near-threshold CPU/GPU discrepancies.
        torch.set_float32_matmul_precision("highest")
        val_scores = score(model, X_val, batch_size)
        torch.set_float32_matmul_precision("high")
        threshold = choose_threshold(y_val, val_scores)
        metrics = evaluate(y_val, val_scores, threshold)
        rank = (metrics["f1"], metrics["precision"])
        if best is None or rank > best["rank"]:
            best = {
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "rank": rank, "threshold": threshold, "metrics": metrics,
                "best_epoch": epoch,
            }
            stale = 0
        else:
            stale += 1
        train_loss = total_loss.item() / len(y_train)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"    epoch {epoch:02d}: loss={train_loss:.5f}; lr={learning_rate:.1e}; "
            f"val F1={metrics['f1']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}", flush=True,
        )
        if scheduler is not None:
            scheduler.step(metrics["f1"])
        if stale >= params["patience"]:
            break

    # Freeze this trial's checkpoint first, then select its threshold using
    # CPU VALIDATION inference, the same inference path used in deployment.
    model = model.cpu()
    model.load_state_dict(best["state_dict"])
    torch.set_float32_matmul_precision("highest")
    val_scores = score(model, X_val.cpu(), batch_size)
    threshold = choose_threshold(y_val, val_scores)
    metrics = evaluate(y_val, val_scores, threshold)
    torch.set_float32_matmul_precision("high")
    best.update(
        threshold=threshold, metrics=metrics, rank=(metrics["f1"], metrics["precision"]),
        epochs_ran=epoch, train_seconds=time.perf_counter() - started,
        config=config, params=params.copy(),
    )
    return best


def main():
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    try:
        from .traditional_ml import read_split, evaluate, DATA_PROTOCOL, LABEL_DEFINITION
    except ImportError:
        from traditional_ml import read_split, evaluate, DATA_PROTOCOL, LABEL_DEFINITION

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparisons, trials = [], []
    versions = {package: version(package) for package in (
        "torch", "rtdl-revisiting-models", "numpy", "pandas", "scikit-learn"
    )}

    with threadpool_limits(limits=4):
        for target in TARGETS:
            horizon = int(target.removeprefix("horizon_label_").removesuffix("min"))
            train_frame, train_y = read_split(target, "train")
            columns = train_frame.columns.tolist()
            val_frame, val_y = read_split(target, "val", columns)
            scaler = StandardScaler().fit(train_frame)
            X_train = torch.from_numpy(standardize(train_frame, scaler.mean_, scaler.scale_)).to(device)
            X_val = torch.from_numpy(standardize(val_frame, scaler.mean_, scaler.scale_)).to(device)
            y_train = torch.tensor(train_y.to_numpy(), dtype=torch.float32, device=device)
            y_val = val_y.to_numpy()
            del train_frame, val_frame
            print(f"\n{target}: train={len(train_y):,}, val={len(val_y):,}, features={len(columns)}", flush=True)

            for name in MODEL_CONFIGS:
                best = None
                first_trial = len(trials)
                settings = parameter_trials(name, horizon)
                for trial_id, params in enumerate(settings, start=1):
                    print(f"  {name}: trial {trial_id}/{len(settings)} {params}", flush=True)
                    result = fit_trial(name, params, X_train, y_train, X_val, y_val, device)
                    trials.append({
                        "target": target, "model": name, "trial_id": trial_id,
                        "params": json.dumps(params, sort_keys=True),
                        "threshold": result["threshold"], "selected": False,
                        "best_epoch": result["best_epoch"], "epochs_ran": result["epochs_ran"],
                        "train_seconds": result["train_seconds"],
                        "evaluation_device": "cpu",
                        **{f"val_{k}": v for k, v in result["metrics"].items()},
                    })
                    if best is None or result["rank"] > best["rank"]:
                        best = {**result, "trial_id": trial_id}
                    pd.DataFrame(trials).to_csv(RESULTS_DIR / "dl_tuning_trials.csv", index=False)

                trials[first_trial + best["trial_id"] - 1]["selected"] = True
                # Read test data only after selecting the model and threshold.
                test_frame, test_y = read_split(target, "test", columns)
                # Evaluate on CPU, matching load_model's deployment default.
                # CUDA arithmetic can shift scores close to the threshold.
                model = make_model(name, len(columns), best["config"])
                model.load_state_dict(best["state_dict"])
                X_test = torch.from_numpy(standardize(test_frame, scaler.mean_, scaler.scale_))
                torch.set_float32_matmul_precision("highest")
                metrics = evaluate(test_y, score(model, X_test, best["params"]["batch_size"]), best["threshold"])
                torch.set_float32_matmul_precision("high")
                artifact = {
                    "format_version": 1, "model_name": name, "model_config": best["config"],
                    "state_dict": best["state_dict"], "target": target,
                    "horizon_min": horizon,
                    "feature_columns": columns,
                    "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float64),
                    "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float64),
                    "threshold": best["threshold"], "score_type": "sigmoid_uncalibrated",
                    "batch_size": best["params"]["batch_size"], "best_params": best["params"],
                    "best_epoch": best["best_epoch"], "max_epochs": best["params"]["max_epochs"],
                    "patience": best["params"]["patience"], "seed": SEED,
                    "versions": versions, "fit_split": "train",
                    "training_device": str(device), "evaluation_device": "cpu",
                    "selection_metric": "validation_positive_f1_then_precision",
                    "threshold_source": "val", "data_protocol": DATA_PROTOCOL,
                    "label_definition": LABEL_DEFINITION,
                    "class_counts": {
                        s: {"positive": int(y.sum()), "negative": int(len(y) - y.sum())}
                        for s, y in (("train", train_y), ("val", val_y), ("test", test_y))
                    },
                    "val_metrics": best["metrics"], "test_metrics": metrics,
                }
                path = MODELS_DIR / f"{target}__{name}.pt"
                torch.save(artifact, path)
                comparisons.append({
                    "target": target, "model": name,
                    "best_params": json.dumps({**best["params"], "best_epoch": best["best_epoch"]}, sort_keys=True),
                    "threshold": best["threshold"], **metrics,
                })
                pd.DataFrame(comparisons).to_csv(RESULTS_DIR / "dl_model_comparison.csv", index=False)
                pd.DataFrame(trials).to_csv(RESULTS_DIR / "dl_tuning_trials.csv", index=False)
                print(f"  Saved {path.name}: test precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}", flush=True)
                del model, X_test, test_frame
            del X_train, X_val, y_train
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nFinished: {len(comparisons)} models, {len(trials)} parameter trials.", flush=True)


if __name__ == "__main__":
    main()
