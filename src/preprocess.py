"""Prepare MetroPT-3 data for the modelling stage.

This is deliberately a small, single-purpose pipeline:

``raw CSV -> labels -> chronological split -> causal window features -> CSV``

Run it from the repository root::

    python src/preprocess.py

The final files are written directly to ``data/processed``. The warning
horizons and history windows are constants below because they are part of the
current experiment definition, not user-specific runtime settings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Current experiment settings.
HORIZONS_MIN = (5, 10, 20, 30)
WINDOWS_MIN = (5, 20, 60)
MIN_HISTORY_OBSERVATIONS = 30
SPLITS = ("train", "val", "test")

FAILURES = (
    ("F1", pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-18 23:59:00")),
    ("F2", pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 06:00:00")),
    ("F3", pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 14:30:00")),
    ("F4", pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),
)

# F3 is validation and F4 is the untouched future test event. The 30-minute
# buffer keeps F3's longest warning window out of the training partition.
TRAIN_END = pd.Timestamp("2020-06-05 09:30:00")
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

# Input and output locations are fixed inside the repository. Keeping these
# paths relative to the repository root makes the same command work for every
# teammate without machine-specific configuration.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPOSITORY_ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"
OUTPUT_DIR = REPOSITORY_ROOT / "data" / "processed"


def make_causal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create current-value and time-based rolling features for one context."""

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
        ["timestamp", "split", "in_failure_window", "failure_id", *TARGET_COLS]
    ].reset_index(drop=True)
    feature_frame = pd.concat([base, pd.DataFrame(features)], axis=1)
    columns = [
        "timestamp",
        "split",
        "in_failure_window",
        "failure_id",
        *TARGET_COLS,
        ready_name,
        *feature_columns,
    ]
    return feature_frame[columns]


def main() -> None:
    """Run the complete preprocessing pipeline."""

    # 1. Load and clean the raw data.
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

    # 2. Add documented failure intervals and the four warning targets.
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

    # 3. Assign the chronological split. The split exists only in memory; the
    # final CSV filename records which partition each row belongs to.
    df["split"] = np.select(
        [df["timestamp"] < TRAIN_END, df["timestamp"] < VAL_END],
        ["train", "val"],
        default="test",
    )
    print("Split sizes:")
    print(df["split"].value_counts().reindex(SPLITS, fill_value=0).to_string())
    for split in SPLITS:
        counts = [
            int(df.loc[df["split"] == split, target].sum()) for target in TARGET_COLS
        ]
        print(f"{split}: positives {counts}")

    # 4. Build causal features and export one final CSV per split. The history
    # tail is carried across boundaries so the first validation/test rows can
    # use legal sensor history without using future observations.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_tail: pd.DataFrame | None = None

    for split in SPLITS:
        current = df.loc[df["split"] == split].sort_values("timestamp").reset_index(drop=True)
        context = (
            pd.concat([history_tail, current], ignore_index=True)
            if history_tail is not None
            else current.copy()
        )
        context_features = make_causal_features(context)
        current_features = context_features.tail(len(current)).drop(columns=["split"])

        # The actual final export: no temporary ETL CSV is written.
        output_path = OUTPUT_DIR / f"{split}.csv"
        current_features.to_csv(output_path, index=False)
        ready = int(current_features[f"history_ready_{max(WINDOWS_MIN)}min"].sum())
        print(
            f"{split}: {len(current_features):,} rows, "
            f"{ready:,} history-ready -> {output_path}"
        )

        # Keep only the maximum-window tail for the next split.
        history_tail = context.loc[
            context["timestamp"]
            >= context["timestamp"].max() - pd.Timedelta(minutes=max(WINDOWS_MIN))
        ].copy()


if __name__ == "__main__":
    main()
