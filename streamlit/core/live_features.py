from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# FIND THE REAL src/preprocess.py
# ============================================================

# Current file:
# project/streamlit/core/live_features.py
#
# We want:
# project/src/preprocess.py
#
# parents[0] = core
# parents[1] = streamlit
# parents[2] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Import the REAL preprocessing code used during training.
from preprocess import (
    make_causal_features,
    SENSOR_VALUE_COLS,
    TARGET_COLS,
    WINDOWS_MIN,
)


READY_COL = f"history_ready_{max(WINDOWS_MIN)}min"

_AUDIT_COLS = {
    "timestamp",
    "in_failure_window",
    "failure_id",
    *TARGET_COLS,
    READY_COL,
}


def compute_latest_features(buffer_rows: list[dict]) -> dict | None:
    """
    Convert the current rolling buffer of raw sensor rows into
    the same features used during model training.

    Returns:
        {
            "is_ready": bool,
            "features": {...}
        }

    Returns None if there is no data.
    """

    if not buffer_rows:
        return None

    # --------------------------------------------------------
    # Convert rolling buffer to DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(buffer_rows)

    if "timestamp" not in df.columns:
        raise ValueError(
            "Each sensor row must contain a 'timestamp' column."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Check required sensor columns
    # --------------------------------------------------------

    missing = [
        col
        for col in SENSOR_VALUE_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required sensor columns: {missing}"
        )

    # --------------------------------------------------------
    # Make sure sensor values are numeric
    # --------------------------------------------------------

    for col in SENSOR_VALUE_COLS:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

    # --------------------------------------------------------
    # Add harmless placeholders required by preprocess.py
    #
    # These are NOT used as model features.
    # --------------------------------------------------------

    df["in_failure_window"] = False

    df["failure_id"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string"
    )

    for col in TARGET_COLS:
        df[col] = 0

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This calls the REAL training feature-engineering
    # function.
    # --------------------------------------------------------

    feature_frame = make_causal_features(df)

    latest = feature_frame.iloc[-1]

    # --------------------------------------------------------
    # Check whether enough history exists
    # --------------------------------------------------------

    is_ready = bool(latest[READY_COL])

    # --------------------------------------------------------
    # Get only model features
    # --------------------------------------------------------

    feature_cols = [
        col
        for col in feature_frame.columns
        if col not in _AUDIT_COLS
    ]

    features = latest[feature_cols].to_dict()

    return {
        "is_ready": is_ready,
        "features": features,
    }