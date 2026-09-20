"""Prepare the four MetroPT-3 modelling datasets: python src/preprocess.py.

Raw data -> warning labels -> causal window features -> per-target SMOTE
and undersampling -> stratified random train/val/test split -> CSV.
Outputs: data/processed/horizon_label_<N>min/{train,val,test}.csv.

Resampling the full pool before randomly splitting overlapping windows can
inflate evaluation scores. These datasets do not measure generalisation to
unseen failure events. Use chronological raw data, not synthetic rows, for replay.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


# Future warning targets and past-data feature windows (minutes).
HORIZONS_MIN = (20, 30, 40, 50)
WINDOWS_MIN = (5, 20, 60)
MIN_HISTORY_OBSERVATIONS = 30  # Records within the largest window, not minutes.

# Rebalancing and stratified random splitting.
RANDOM_SEED = 42
SMOTE_POSITIVE_MULTIPLIER = 2.0  # Final positive count is twice the original.
NEGATIVES_PER_POSITIVE = 600    # Keep up to 600 real negatives per positive.
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test

FAILURES = (
    ("F1", pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-18 23:59:00")),
    ("F2", pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 06:00:00")),
    ("F3", pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 14:30:00")),
    ("F4", pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),
)

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

# Input and output locations are fixed inside the repository. Keeping these
# paths relative to the repository root makes the same command work for every
# teammate without machine-specific configuration.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPOSITORY_ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"
OUTPUT_DIR = REPOSITORY_ROOT / "data" / "processed"


def make_causal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the unchanged 173 features from current/past rows in time order."""

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

    # Internal quality gate; filtered and removed before SMOTE/export.
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
    """Load raw readings, sort by time, and label the four warning horizons."""

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


def rebalance_and_save(pool_features: pd.DataFrame) -> None:
    """Filter history, rebalance each target, split randomly, and save three CSVs."""

    print("\nPreparing rebalanced datasets")

    ready_col = f"history_ready_{max(WINDOWS_MIN)}min"
    # Apply the minimum-history filter before resampling.
    pool = pool_features.loc[pool_features[ready_col] == 1].reset_index(drop=True)
    print(f"Pool after {ready_col} filter: {len(pool):,} of {len(pool_features):,} rows")

    # Audit fields, readiness and ALL target labels must stay out of X.
    non_feature_cols = {"timestamp", "in_failure_window", "failure_id", ready_col, *TARGET_COLS}
    feature_cols = [c for c in pool.columns if c not in non_feature_cols]

    for horizon in HORIZONS_MIN:
        target_col = f"horizon_label_{horizon}min"
        print(f"\n--- Target: {target_col} ---")

        X = pool[feature_cols].copy()
        y = pool[target_col].copy()

        n_pos_before = int(y.sum())
        n_neg_before = int((y == 0).sum())
        print(f"Before SMOTE: positive={n_pos_before:,}, negative={n_neg_before:,} "
              f"(ratio 1:{n_neg_before / max(n_pos_before, 1):.1f})")

        target_pos_count = int(round(n_pos_before * SMOTE_POSITIVE_MULTIPLIER))
        smote = SMOTE(sampling_strategy={1: target_pos_count}, random_state=RANDOM_SEED)
        X_smote, y_smote = smote.fit_resample(X, y)
        n_pos_after_smote = int((y_smote == 1).sum())
        n_neg_after_smote = int((y_smote == 0).sum())
        print(f"After SMOTE:  positive={n_pos_after_smote:,}, negative={n_neg_after_smote:,}")

        target_neg_count = min(n_neg_after_smote, n_pos_after_smote * NEGATIVES_PER_POSITIVE)
        rus = RandomUnderSampler(sampling_strategy={0: target_neg_count}, random_state=RANDOM_SEED)
        X_res, y_res = rus.fit_resample(X_smote, y_smote)
        n_pos_final = int((y_res == 1).sum())
        n_neg_final = int((y_res == 0).sum())
        print(f"After undersample: positive={n_pos_final:,}, negative={n_neg_final:,} "
              f"(ratio 1:{n_neg_final / max(n_pos_final, 1):.1f})")

        resampled = X_res.copy()
        resampled[target_col] = y_res.to_numpy()
        resampled = resampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        train_frac, val_frac, test_frac = SPLIT_RATIOS
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


def main() -> None:
    """Load, label, build features, and export the four rebalanced datasets."""

    labelled = load_and_label()
    pool_features = make_causal_features(labelled)

    rebalance_and_save(pool_features)


if __name__ == "__main__":
    main()
