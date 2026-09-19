"""Prepare MetroPT-3 data for the modelling stage.

Two outputs are produced from ONE shared feature pool:

1. The original chronological pipeline (unchanged behaviour):
   ``raw CSV -> labels -> chronological split -> causal window features -> CSV``
   Written to ``data/processed/{train,val,test}.csv``.

2. A new, separate, TARGET-SPECIFIC rebalanced pipeline requested for
   experimentation with SMOTE + undersampling:
   ``raw CSV -> labels -> causal window features (whole pool) -> per-target
   SMOTE oversample -> per-target random undersample -> shuffle -> random
   train/val/test split -> CSV``
   Written to ``data/processed/horizon_label_<N>min/{train,val,test}.csv``
   for N in {20, 30, 40, 50}.

*** IMPORTANT METHODOLOGICAL NOTE ***
Pipeline 2 uses SMOTE + random undersampling on the FULL pool BEFORE a
RANDOM (non-chronological) train/val/test split. This is a deliberate
departure from every leakage-prevention rule documented in
docs/metadata.md and docs/preprocess_document.md for pipeline 1:
  - SMOTE fit before splitting can let synthetic train rows be near-
    duplicates of real rows that end up in val/test (SMOTE interpolates
    between real neighbours).
  - A random split after rolling-window feature engineering scatters
    heavily-overlapping adjacent windows across splits.
Both effects can inflate validation/test metrics in pipeline 2 in a way
that does not reflect real early-warning performance. Pipeline 1's
chronological, event-aware split is unaffected and remains the
methodologically sound evaluation. Treat pipeline 2's outputs as an
experimental/exploratory artifact, not the project's primary evaluation
set, unless the team has an explicit, discussed reason to accept the
tradeoff (e.g. a specific assignment requirement).

Run from the repository root::

    python src/preprocess.py

The final files are written directly to ``data/processed``. The warning
horizons and history windows are constants below because they are part of the
current experiment definition, not user-specific runtime settings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


# Current experiment settings.
# Extended from (5, 10, 20, 30) to also cover the 40/50-minute horizons
# needed by the new rebalanced pipeline, while keeping the original 5/10min
# horizons intact for the existing chronological pipeline.
HORIZONS_MIN = (5, 10, 20, 30, 40, 50)
WINDOWS_MIN = (5, 20, 60)
MIN_HISTORY_OBSERVATIONS = 30
SPLITS = ("train", "val", "test")

# Horizons processed by the new SMOTE-rebalanced, per-target pipeline.
REBALANCE_HORIZONS_MIN = (20, 30, 40, 50)
RANDOM_SEED = 42
SMOTE_POSITIVE_MULTIPLIER = 2.0   # SMOTE brings the positive class to ~2x its original count
TARGET_POS_NEG_RATIO = 5          # after undersampling, negatives are ~5x positives (1:5)
REBALANCE_SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train/val/test, NOT specified in the request -- chosen
                                              # here as a reasonable default; confirm with the team.

FAILURES = (
    ("F1", pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-18 23:59:00")),
    ("F2", pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 06:00:00")),
    ("F3", pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 14:30:00")),
    ("F4", pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),
)

# F3 is validation and F4 is the untouched future test event (pipeline 1 only).
# The 30-minute buffer keeps F3's longest warning window out of the training
# partition. NOTE: this buffer was sized for the original 30-minute max
# horizon; it no longer covers the new 40/50-minute horizons. This only
# matters for pipeline 1 (which still only reports the original 5/10/20/30min
# targets in its class-count printout); pipeline 2 does not use this
# chronological boundary at all.
# The train/val boundary must sit at least max(HORIZONS_MIN) before the
# validation event (F3), otherwise rows labelled positive for F3 fall inside
# the TRAIN partition and leak the validation event's warning window.
# This is derived rather than hardcoded: the original fixed 09:30 value was a
# 30-minute buffer, which silently became a leak once the 40/50-minute
# horizons were added.
_F3_START = FAILURES[2][1]
TRAIN_END = _F3_START - pd.Timedelta(minutes=max(HORIZONS_MIN))
VAL_END = pd.Timestamp("2020-07-14 00:00:00")

DIGITAL_COLS = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)

SENSOR_COLS = (
    "timestamp",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    *DIGITAL_COLS,
)
SENSOR_VALUE_COLS = tuple(column for column in SENSOR_COLS if column != "timestamp")
ANALOG_COLS = tuple(column for column in SENSOR_VALUE_COLS if column not in DIGITAL_COLS)
TARGET_COLS = tuple(f"horizon_label_{minutes}min" for minutes in HORIZONS_MIN)
# Only the original 4 horizons, for pipeline 1's unchanged printout/behaviour.
ORIGINAL_TARGET_COLS = tuple(f"horizon_label_{m}min" for m in (5, 10, 20, 30))

# Input and output locations are fixed inside the repository. Keeping these
# paths relative to the repository root makes the same command work for every
# teammate without machine-specific configuration.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPOSITORY_ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"
OUTPUT_DIR = REPOSITORY_ROOT / "data" / "processed"


def make_causal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create current-value and time-based rolling features for one context.

    Unchanged from the original pipeline. Because every feature here is
    strictly backward-looking (``closed="both"``, rolling over past rows
    only), it is safe to call this ONCE on the full, continuously-ordered
    dataset -- a row's features never depend on which split (if any) it will
    later be assigned to.
    """

    indexed = data.set_index("timestamp", drop=False)
    features: dict[str, np.ndarray] = {}
    feature_columns: list[str] = []

    # Latest readings available at this timestamp.
    for column in SENSOR_VALUE_COLS:
        name = f"current_{column}"
        features[name] = data[column].to_numpy(dtype=np.float32)
        feature_columns.append(name)

    # The source cadence is nominally 10 seconds but has long gaps.
    interval = data["timestamp"].diff().dt.total_seconds().fillna(0.0)
    features["sampling_interval_sec"] = interval.to_numpy(dtype=np.float32)
    features["gap_over_60sec"] = (interval > 60.0).to_numpy(dtype=np.int8)
    feature_columns += ["sampling_interval_sec", "gap_over_60sec"]

    indexed["_row_count"] = 1.0
    transitions = indexed[list(DIGITAL_COLS)].diff().abs().fillna(0.0)

    for minutes in WINDOWS_MIN:
        window = f"{minutes}min"
        analogue = indexed[list(ANALOG_COLS)].rolling(
            window=window, min_periods=1, closed="both"
        )
        means = analogue.mean()
        stds = analogue.std().fillna(0.0)
        minimums = analogue.min()
        maximums = analogue.max()

        for column in ANALOG_COLS:
            values = {
                f"{column}_mean_{window}": means[column],
                f"{column}_std_{window}": stds[column],
                f"{column}_min_{window}": minimums[column],
                f"{column}_max_{window}": maximums[column],
                f"{column}_range_{window}": maximums[column] - minimums[column],
            }
            for name, series in values.items():
                features[name] = series.to_numpy(dtype=np.float32, na_value=np.nan)
                feature_columns.append(name)

        digital = indexed[list(DIGITAL_COLS)].rolling(
            window=window, min_periods=1, closed="both"
        )
        digital_means = digital.mean()
        digital_transitions = transitions.rolling(
            window=window, min_periods=1, closed="both"
        ).sum()
        for column in DIGITAL_COLS:
            mean_name = f"{column}_mean_{window}"
            transition_name = f"{column}_transitions_{window}"
            features[mean_name] = digital_means[column].to_numpy(
                dtype=np.float32, na_value=np.nan
            )
            features[transition_name] = digital_transitions[column].to_numpy(
                dtype=np.float32, na_value=np.nan
            )
            feature_columns += [mean_name, transition_name]

        count_name = f"history_observations_{window}"
        features[count_name] = indexed["_row_count"].rolling(
            window=window, min_periods=1, closed="both"
        ).sum().to_numpy(dtype=np.float32)
        feature_columns.append(count_name)

    # This is a quality flag used to filter rows before modelling, not a model
    # input. It is kept in the output for transparent data-quality reporting.
    ready_name = f"history_ready_{max(WINDOWS_MIN)}min"
    features[ready_name] = (
        features[f"history_observations_{max(WINDOWS_MIN)}min"]
        >= MIN_HISTORY_OBSERVATIONS
    ).astype(np.int8)

    base = data[
        ["timestamp", "in_failure_window", "failure_id", *TARGET_COLS]
    ].reset_index(drop=True)
    feature_frame = pd.concat([base, pd.DataFrame(features)], axis=1)
    columns = [
        "timestamp",
        "in_failure_window",
        "failure_id",
        *TARGET_COLS,
        ready_name,
        *feature_columns,
    ]
    return feature_frame[columns]


def load_and_label() -> pd.DataFrame:
    """Steps 1-2 of the original pipeline: load, clean, and label. Unchanged."""

    df = pd.read_csv(INPUT_PATH)
    df.columns = [str(column).strip() for column in df.columns]
    if df.columns[0] != "timestamp":
        df = df.drop(columns=[df.columns[0]])
    missing = [column for column in SENSOR_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise").astype(
        "datetime64[ns]"
    )
    for column in SENSOR_VALUE_COLS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Raw rows: {len(df):,}")
    print(f"Timestamp range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    gaps = df["timestamp"].diff().dt.total_seconds().dropna()
    print(f"Gaps over 60 seconds: {(gaps > 60).sum():,}")

    df["in_failure_window"] = False
    df["failure_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
    for failure_id, start, end in FAILURES:
        inside = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df.loc[inside, "in_failure_window"] = True
        df.loc[inside, "failure_id"] = failure_id

    starts_ns = np.array([start.value for _, start, _ in FAILURES], dtype=np.int64)
    timestamps_ns = df["timestamp"].astype("int64").to_numpy()
    next_position = np.searchsorted(starts_ns, timestamps_ns, side="right")
    has_next = next_position < len(starts_ns)
    next_start_ns = np.full(len(df), np.nan, dtype=np.float64)
    next_start_ns[has_next] = starts_ns[next_position[has_next]]
    minutes_to_next = (next_start_ns - timestamps_ns) / (60 * 1e9)
    valid_warning_row = (
        ~df["in_failure_window"]
        & np.isfinite(minutes_to_next)
        & (minutes_to_next > 0)
    )
    for horizon in HORIZONS_MIN:
        df[f"horizon_label_{horizon}min"] = (
            valid_warning_row & (minutes_to_next <= horizon)
        ).astype(np.int8)

    return df


def run_chronological_pipeline(labelled: pd.DataFrame, pool_features: pd.DataFrame) -> None:
    """Pipeline 1 (original, unchanged behaviour): chronological split, event-aware,
    written to data/processed/{train,val,test}.csv. Derived from the SAME
    pool_features computed once over the whole dataset -- safe because every
    feature is backward-looking, so slicing by split afterward is equivalent
    to the original tail-carry approach, just computed once instead of
    per-split.
    """

    split = np.select(
        [labelled["timestamp"] < TRAIN_END, labelled["timestamp"] < VAL_END],
        ["train", "val"],
        default="test",
    )
    print("\n[Pipeline 1: chronological] Split sizes:")
    split_series = pd.Series(split, index=labelled.index)
    print(split_series.value_counts().reindex(SPLITS, fill_value=0).to_string())
    for s in SPLITS:
        counts = [
            int(labelled.loc[split_series == s, target].sum())
            for target in ORIGINAL_TARGET_COLS
        ]
        print(f"{s}: positives {dict(zip(ORIGINAL_TARGET_COLS, counts))}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        mask = (split_series == s).to_numpy()
        current_features = pool_features.loc[mask].reset_index(drop=True)
        output_path = OUTPUT_DIR / f"{s}.csv"
        current_features.to_csv(output_path, index=False)
        ready = int(current_features[f"history_ready_{max(WINDOWS_MIN)}min"].sum())
        print(
            f"[Pipeline 1] {s}: {len(current_features):,} rows, "
            f"{ready:,} history-ready -> {output_path}"
        )


def run_rebalanced_pipeline(pool_features: pd.DataFrame) -> None:
    """Pipeline 2 (new): per-target SMOTE oversample + random undersample +
    random shuffle + random train/val/test split.

    See the METHODOLOGICAL NOTE in the module docstring -- this pipeline is
    intentionally non-chronological, per the request that introduced it, and
    should be treated as experimental rather than the project's primary
    evaluation methodology.
    """

    print("\n" + "=" * 70)
    print("[Pipeline 2: SMOTE-rebalanced, per-target] Starting")
    print("=" * 70)

    ready_col = f"history_ready_{max(WINDOWS_MIN)}min"
    # Apply the project's established quality gate before doing anything else
    # (docs/metadata.md: "Apply the history_ready_60min filter before fitting").
    pool = pool_features.loc[pool_features[ready_col] == 1].reset_index(drop=True)
    print(f"Pool after history_ready_60min filter: {len(pool):,} of {len(pool_features):,} rows")

    # Columns that are never legitimate model features, consistent with
    # docs/metadata.md's rule (timestamp/failure_id/in_failure_window +
    # target columns + the quality gate itself; "split indicators" per the
    # new request map onto these same audit columns since pipeline 2 has no
    # chronological split column).
    non_feature_cols = {"timestamp", "in_failure_window", "failure_id", ready_col, *TARGET_COLS}
    feature_cols = [c for c in pool.columns if c not in non_feature_cols]

    for horizon in REBALANCE_HORIZONS_MIN:
        target_col = f"horizon_label_{horizon}min"
        print(f"\n--- Target: {target_col} ---")

        X = pool[feature_cols].copy()
        y = pool[target_col].copy()

        n_pos_before = int(y.sum())
        n_neg_before = int((y == 0).sum())
        print(f"Before SMOTE: positive={n_pos_before:,}, negative={n_neg_before:,} "
              f"(ratio 1:{n_neg_before / max(n_pos_before, 1):.1f})")

        if n_pos_before < 2:
            print(f"  SKIPPED: fewer than 2 positive rows for {target_col} -- "
                  f"SMOTE cannot interpolate. Check the horizon/failure data for this target.")
            continue

        target_pos_count = int(round(n_pos_before * SMOTE_POSITIVE_MULTIPLIER))
        smote = SMOTE(sampling_strategy={1: target_pos_count}, random_state=RANDOM_SEED)
        X_smote, y_smote = smote.fit_resample(X, y)
        n_pos_after_smote = int((y_smote == 1).sum())
        n_neg_after_smote = int((y_smote == 0).sum())
        print(f"After SMOTE:  positive={n_pos_after_smote:,}, negative={n_neg_after_smote:,}")

        target_neg_count = min(n_neg_after_smote, n_pos_after_smote * TARGET_POS_NEG_RATIO)
        rus = RandomUnderSampler(sampling_strategy={0: target_neg_count}, random_state=RANDOM_SEED)
        X_res, y_res = rus.fit_resample(X_smote, y_smote)
        n_pos_final = int((y_res == 1).sum())
        n_neg_final = int((y_res == 0).sum())
        print(f"After undersample: positive={n_pos_final:,}, negative={n_neg_final:,} "
              f"(ratio 1:{n_neg_final / max(n_pos_final, 1):.1f})")

        resampled = X_res.copy()
        resampled[target_col] = y_res.to_numpy()
        resampled = resampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        train_frac, val_frac, test_frac = REBALANCE_SPLIT_RATIOS
        train_df, temp_df = train_test_split(
            resampled, train_size=train_frac, random_state=RANDOM_SEED,
            stratify=resampled[target_col],
        )
        relative_val_frac = val_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(
            temp_df, train_size=relative_val_frac, random_state=RANDOM_SEED,
            stratify=temp_df[target_col],
        )

        target_dir = OUTPUT_DIR / target_col
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
            split_df = split_df.reset_index(drop=True)
            null_count = int(split_df.isna().sum().sum())
            if null_count:
                raise ValueError(
                    f"{target_col}/{name}.csv would contain {null_count} null values -- "
                    f"refusing to write. Investigate the source feature(s) before rerunning."
                )
            out_path = target_dir / f"{name}.csv"
            split_df.to_csv(out_path, index=False)
            pos = int(split_df[target_col].sum())
            neg = len(split_df) - pos
            print(f"  {target_col}/{name}.csv: {len(split_df):,} rows "
                  f"(positive={pos:,}, negative={neg:,}) -> {out_path}")


def run_chronological_smote_pipeline(labelled: pd.DataFrame, pool_features: pd.DataFrame) -> None:
    """Pipeline 3 (recommended): chronological split FIRST, then SMOTE +
    undersampling applied to the TRAIN fold ONLY.

    This keeps the benefit the rebalancing was introduced for -- a classifier
    cannot learn from ~50 positive rows in 220k -- without contaminating the
    evaluation:

      * val and test keep their real, untouched, naturally-imbalanced rows,
        so precision/recall/F1 mean what they normally mean;
      * no synthetic row derived from a val/test neighbour can ever reach
        the training set, because resampling happens after the split;
      * the chronological order is preserved, so test remains a genuinely
        future, never-seen failure event (F4).

    Written to ``data/processed/chrono_horizon_label_<N>min/{train,val,test}.csv``.
    """

    print("\n" + "=" * 70)
    print("[Pipeline 3: chronological split + train-only SMOTE] Starting")
    print("=" * 70)

    ready_col = f"history_ready_{max(WINDOWS_MIN)}min"
    split = np.select(
        [labelled["timestamp"] < TRAIN_END, labelled["timestamp"] < VAL_END],
        ["train", "val"],
        default="test",
    )
    split_series = pd.Series(split, index=labelled.index)

    pool = pool_features.copy()
    pool["_split"] = split_series.to_numpy()
    pool = pool.loc[pool[ready_col] == 1].reset_index(drop=True)

    non_feature_cols = {
        "timestamp", "in_failure_window", "failure_id", ready_col, "_split", *TARGET_COLS
    }
    feature_cols = [c for c in pool.columns if c not in non_feature_cols]

    for horizon in REBALANCE_HORIZONS_MIN:
        target_col = f"horizon_label_{horizon}min"
        print(f"\n--- Target: {target_col} ---")

        parts = {s: pool.loc[pool["_split"] == s] for s in SPLITS}
        for s in SPLITS:
            y_s = parts[s][target_col]
            print(f"  {s}: {len(parts[s]):,} rows, positive={int(y_s.sum()):,} "
                  f"({y_s.mean()*100:.4f}%)")

        X_tr = parts["train"][feature_cols]
        y_tr = parts["train"][target_col]
        n_pos = int(y_tr.sum())

        if n_pos < 2:
            print(f"  SKIPPED: only {n_pos} positive training row(s) -- SMOTE needs >= 2.")
            continue

        k_neighbors = min(5, n_pos - 1)
        if k_neighbors < 5:
            print(f"  NOTE: only {n_pos} positive training rows -- reducing SMOTE "
                  f"k_neighbors to {k_neighbors}. Synthetic samples will have very "
                  f"low diversity; treat any gain with caution.")

        target_pos = int(round(n_pos * SMOTE_POSITIVE_MULTIPLIER))
        smote = SMOTE(sampling_strategy={1: target_pos}, k_neighbors=k_neighbors,
                      random_state=RANDOM_SEED)
        X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
        print(f"  TRAIN after SMOTE: positive={int((y_sm==1).sum()):,}, "
              f"negative={int((y_sm==0).sum()):,}")

        n_pos_sm = int((y_sm == 1).sum())
        n_neg_sm = int((y_sm == 0).sum())
        target_neg = min(n_neg_sm, n_pos_sm * TARGET_POS_NEG_RATIO)
        rus = RandomUnderSampler(sampling_strategy={0: target_neg}, random_state=RANDOM_SEED)
        X_res, y_res = rus.fit_resample(X_sm, y_sm)
        print(f"  TRAIN after undersample: positive={int((y_res==1).sum()):,}, "
              f"negative={int((y_res==0).sum()):,} "
              f"(1:{int((y_res==0).sum())/max(int((y_res==1).sum()),1):.1f})")

        train_out = X_res.copy()
        train_out[target_col] = y_res.to_numpy()
        train_out = train_out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        target_dir = OUTPUT_DIR / f"chrono_{target_col}"
        target_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "train": train_out,
            "val": parts["val"][[*feature_cols, target_col]].reset_index(drop=True),
            "test": parts["test"][[*feature_cols, target_col]].reset_index(drop=True),
        }
        for name, frame in outputs.items():
            nulls = int(frame.isna().sum().sum())
            if nulls:
                raise ValueError(
                    f"chrono_{target_col}/{name}.csv would contain {nulls} nulls -- refusing to write."
                )
            out_path = target_dir / f"{name}.csv"
            frame.to_csv(out_path, index=False)
            pos = int(frame[target_col].sum())
            print(f"  chrono_{target_col}/{name}.csv: {len(frame):,} rows "
                  f"(positive={pos:,}, negative={len(frame)-pos:,}) -> {out_path}")


def main() -> None:
    """Run both pipelines from one shared, once-computed feature pool."""

    labelled = load_and_label()
    pool_features = make_causal_features(labelled)

    run_chronological_pipeline(labelled, pool_features)
    run_rebalanced_pipeline(pool_features)
    run_chronological_smote_pipeline(labelled, pool_features)


if __name__ == "__main__":
    main()
