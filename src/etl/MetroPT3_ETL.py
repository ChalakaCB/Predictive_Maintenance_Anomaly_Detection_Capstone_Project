"""Build the MetroPT-3 ETL dataset used by the modelling notebooks.

The notebook (``MetroPT3_ETL.ipynb``) is intentionally explanatory.  This
script contains the same ETL as a repeatable local command: it reads the raw
MetroPT-3 CSV, adds the documented failure metadata, creates the 5/10/20/30
minute warning labels, creates the notebook's adaptive statistical diagnostics,
performs the chronological F4 hold-out split, and writes train/test CSV files.

Example
-------
    python MetroPT3_ETL.py

The defaults are relative to this file, so the command can be run directly
from ``D:\\pump\\metropt+3+dataset`` without Google Drive or Colab.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (5, 10, 20, 30)
DEFAULT_SPLIT_BOUNDARY = pd.Timestamp("2020-07-14 00:00:00")

FAILURES = (
    ("F1", pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-18 23:59:00")),
    ("F2", pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 06:00:00")),
    ("F3", pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 14:30:00")),
    ("F4", pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),
)

DIGITAL_COLS = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

SENSOR_COLS = [
    "timestamp",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    *DIGITAL_COLS,
]


def load_raw(input_path: Path) -> pd.DataFrame:
    """Load, clean, parse, and chronologically sort the raw CSV."""

    if not input_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = [str(column).strip() for column in df.columns]

    # The downloaded file contains a leftover row index (usually named
    # ``Unnamed: 0``).  Keep the notebook's behaviour, but do not drop the
    # timestamp if a future copy of the file already has no index column.
    first_column = df.columns[0]
    if first_column != "timestamp":
        df = df.drop(columns=[first_column])

    missing_columns = [column for column in SENSOR_COLS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns are missing from the raw CSV: {missing_columns}")

    # Force nanosecond precision so timestamp arithmetic has the same unit as
    # pandas Timestamp.value below (recent pandas versions may infer microseconds
    # when parsing this CSV).
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise").astype(
        "datetime64[ns]"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def print_quality_checks(df: pd.DataFrame) -> None:
    """Print lightweight checks that are useful when auditing a new download."""

    print(f"Raw shape after cleanup: {df.shape}")
    print(f"Timestamp range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print("Missing values:")
    print(df.isna().sum().to_string())

    print("\nDigital-column unique values:")
    for column in DIGITAL_COLS:
        print(f"  {column}: {sorted(df[column].dropna().unique().tolist())}")

    deltas = df["timestamp"].diff().dropna().dt.total_seconds()
    if not deltas.empty:
        print(
            "\nTimestamp spacing (seconds): "
            f"median={deltas.median():g}, min={deltas.min():g}, max={deltas.max():g}, "
            f"gaps >60s={(deltas > 60).sum():,}"
        )


def add_failure_and_horizon_labels(
    df: pd.DataFrame, horizons: Sequence[int]
) -> pd.DataFrame:
    """Add documented failure windows and fixed future-warning targets.

    A row is positive for ``horizon_label_Xmin`` when the next documented
    failure starts within X minutes, the row is strictly before that start,
    and the row is not already inside any failure window.
    """

    df["in_failure_window"] = False
    df["failure_id"] = pd.Series(pd.NA, index=df.index, dtype="string")

    for failure_id, start, end in FAILURES:
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df.loc[mask, "in_failure_window"] = True
        df.loc[mask, "failure_id"] = failure_id

    # Vectorised equivalent of the notebook's ``time_to_next`` function.
    # ``side='right'`` enforces the strict ``failure_start > timestamp`` rule.
    failure_start_ns = np.array([start.value for _, start, _ in FAILURES], dtype=np.int64)
    timestamp_ns = df["timestamp"].astype("int64").to_numpy()
    next_positions = np.searchsorted(failure_start_ns, timestamp_ns, side="right")
    has_next_failure = next_positions < len(failure_start_ns)
    next_failure_ns = np.full(len(df), np.nan, dtype=np.float64)
    next_failure_ns[has_next_failure] = failure_start_ns[next_positions[has_next_failure]]
    df["time_to_next_failure_min"] = (next_failure_ns - timestamp_ns) / (60 * 1e9)

    valid_pre_failure = (
        df["time_to_next_failure_min"].notna()
        & (df["time_to_next_failure_min"] > 0)
        & (~df["in_failure_window"])
    )

    for horizon in horizons:
        label_column = f"horizon_label_{horizon}min"
        df[label_column] = (
            valid_pre_failure & (df["time_to_next_failure_min"] <= horizon)
        ).astype("int8")

    print("\nFailure rows by event:")
    print(df.loc[df["in_failure_window"], "failure_id"].value_counts().sort_index().to_string())
    print("\nHorizon label summary:")
    for horizon in horizons:
        label_column = f"horizon_label_{horizon}min"
        positives = int(df[label_column].sum())
        ratio = 100 * float(df[label_column].mean())
        print(f"  {label_column}: {positives:,} positives ({ratio:.4f}%)")

    return df


def add_adaptive_statistical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's adaptive statistical diagnostics.

    ``oil_temp_anomaly`` is based on a two-hour look-back z-score.  The
    ``off_duration_anomaly`` implementation follows the notebook and marks a
    completed long off-stretch after comparing it with previous stretches.
    It is useful for exploratory analysis, but because a full stretch end is
    used it must not be treated as a causal online feature without redesign.
    """

    # Part 1: oil temperature deviation from its recent history.
    window = 720
    roll_mean = (
        df["Oil_temperature"].rolling(window=window, min_periods=100).mean().shift(1)
    )
    roll_std = (
        df["Oil_temperature"].rolling(window=window, min_periods=100).std().shift(1)
    )
    df["oil_temp_zscore"] = (df["Oil_temperature"] - roll_mean) / roll_std
    df["oil_temp_anomaly"] = (df["oil_temp_zscore"].abs() > 3).astype("int8")

    # Part 2: unusually long motor-off stretches.
    df["is_off"] = df["Motor_current"] < 0.1
    df["stretch_id"] = (df["is_off"] != df["is_off"].shift()).cumsum()

    stretches = df.groupby("stretch_id", sort=True).agg(
        is_off=("is_off", "first"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
    )
    stretches["duration_min"] = (
        (stretches["end"] - stretches["start"]).dt.total_seconds() / 60 + (10 / 60)
    )

    off_stretches = stretches[stretches["is_off"]].copy()
    off_stretches["baseline_mean"] = (
        off_stretches["duration_min"].rolling(20, min_periods=5).mean().shift(1)
    )
    off_stretches["baseline_std"] = (
        off_stretches["duration_min"].rolling(20, min_periods=5).std().shift(1)
    )
    off_stretches["duration_zscore"] = (
        off_stretches["duration_min"] - off_stretches["baseline_mean"]
    ) / off_stretches["baseline_std"]
    off_stretches["duration_anomaly"] = (
        off_stretches["duration_zscore"] > 3
    ).astype("int8")

    anomaly_stretch_ids = set(
        off_stretches.index[off_stretches["duration_anomaly"] == 1]
    )
    df["off_duration_anomaly"] = df["stretch_id"].isin(anomaly_stretch_ids).astype("int8")
    df["adaptive_statistical_label"] = (
        (df["oil_temp_anomaly"] == 1) | (df["off_duration_anomaly"] == 1)
    ).astype("int8")

    df.drop(columns=["is_off", "stretch_id"], inplace=True)

    print("\nAdaptive statistical diagnostics:")
    print(f"  Oil-temperature anomaly rows: {int(df['oil_temp_anomaly'].sum()):,}")
    print(f"  Off-duration anomaly rows: {int(df['off_duration_anomaly'].sum()):,}")
    print(
        "  Combined adaptive label rows: "
        f"{int(df['adaptive_statistical_label'].sum()):,} "
        f"({100 * float(df['adaptive_statistical_label'].mean()):.4f}%)"
    )
    return df


def split_and_save(
    df: pd.DataFrame, output_dir: Path, split_boundary: pd.Timestamp, horizons: Sequence[int]
) -> tuple[Path, Path]:
    """Hold out the latest failure (F4) and write train/test CSV files."""

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["split"] = np.where(df["timestamp"] < split_boundary, "train", "test")

    print("\nChronological split:")
    print(df["split"].value_counts().to_string())
    for split_name in ("train", "test"):
        failures = sorted(
            df.loc[df["split"] == split_name, "failure_id"].dropna().unique().tolist()
        )
        print(f"Failures in {split_name.upper()}: {failures}")
        for horizon in horizons:
            label_column = f"horizon_label_{horizon}min"
            positives = int(df.loc[df["split"] == split_name, label_column].sum())
            print(f"  {label_column} positives: {positives:,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "metropt3_train.csv"
    test_path = output_dir / "metropt3_test.csv"

    train_df = df.loc[df["split"] == "train"].drop(columns=["split"])
    test_df = df.loc[df["split"] == "test"].drop(columns=["split"])
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain saved: {train_path} {train_df.shape}")
    print(f"Test saved:  {test_path} {test_df.shape}")
    return train_path, test_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        default=script_dir / "MetroPT3(AirCompressor).csv",
        help="Path to the raw MetroPT-3 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "ETL_OUTPUT Dataset",
        help="Directory for metropt3_train.csv and metropt3_test.csv.",
    )
    parser.add_argument(
        "--split-boundary",
        type=str,
        default=DEFAULT_SPLIT_BOUNDARY.strftime("%Y-%m-%d %H:%M:%S"),
        help="Rows before this timestamp go to train; later rows go to test.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Warning horizons in minutes (default: 5 10 20 30).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    horizons = tuple(sorted(set(args.horizons)))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("--horizons must contain positive minute values")

    split_boundary = pd.Timestamp(args.split_boundary)
    print(f"Input: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Horizons: {list(horizons)} minutes")
    print(f"Split boundary: {split_boundary}")

    df = load_raw(args.input_path)
    print_quality_checks(df)
    df = add_failure_and_horizon_labels(df, horizons)
    df = add_adaptive_statistical_labels(df)
    split_and_save(df, args.output_dir, split_boundary, horizons)


if __name__ == "__main__":
    main()
