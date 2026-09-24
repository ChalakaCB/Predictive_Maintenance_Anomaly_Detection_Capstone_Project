from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import streamlit as st

from core.state import SessionState
from core.model_registry import registry


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Pump Predictive Maintenance",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MetroPT3(AirCompressor).csv"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "src"
    / "models"
)


# ============================================================
# CONSTANTS
# ============================================================

HORIZONS = [20, 30, 40, 50]

SPEED_OPTIONS = {
    "1×": 1,
    "5×": 5,
    "10×": 10,
    "25×": 25,
    "50×": 50,
    "100×": 100,
}

ALERT_COLORS = {
    "NORMAL": "#2ecc71",
    "WATCH": "#f1c40f",
    "WARNING": "#e67e22",
    "ALERT": "#e74c3c",
}


# ============================================================
# SESSION STATE INITIALISATION
# ============================================================

if "models_loaded_horizon" not in st.session_state:
    st.session_state.models_loaded_horizon = None

if "session" not in st.session_state:
    st.session_state.session = SessionState(
        window_size=400
    )

if "rows" not in st.session_state:
    st.session_state.rows = None

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

if "replay_start" not in st.session_state:
    st.session_state.replay_start = 0

if "replay_end" not in st.session_state:
    st.session_state.replay_end = None

if "is_replaying" not in st.session_state:
    st.session_state.is_replaying = False

if "event_log" not in st.session_state:
    st.session_state.event_log = []

if "active_model_id" not in st.session_state:
    st.session_state.active_model_id = None

if "mode" not in st.session_state:
    st.session_state.mode = "Demo Mode"

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


session = st.session_state.session


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dataset(path: str):
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            "The dataset must contain a timestamp column."
        )

    return df


def load_uploaded_dataset(uploaded_file):
    df = pd.read_csv(uploaded_file)

    if "timestamp" not in df.columns:
        raise ValueError(
            "Uploaded CSV must contain a 'timestamp' column."
        )

    return df


# ============================================================
# MODEL LOADING
# ============================================================

def ensure_models_loaded(horizon_min: int):
    if (
        st.session_state.models_loaded_horizon
        == horizon_min
    ):
        return

    registry.load_all_real_models(
        models_dir=str(MODELS_DIR),
        horizon_min=horizon_min,
    )

    st.session_state.models_loaded_horizon = horizon_min


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Controls")


# ------------------------------------------------------------
# Mode
# ------------------------------------------------------------

mode = st.sidebar.radio(
    "Mode",
    [
        "Full Dataset",
        "Demo Mode",
        "Upload & Test",
    ],
    index=[
        "Full Dataset",
        "Demo Mode",
        "Upload & Test",
    ].index(st.session_state.mode),
)

st.session_state.mode = mode


# ------------------------------------------------------------
# Prediction Horizon
# ------------------------------------------------------------

horizon_min = st.sidebar.selectbox(
    "Prediction Horizon",
    options=HORIZONS,
    index=HORIZONS.index(30),
    format_func=lambda x: f"{x} minutes",
)


# ------------------------------------------------------------
# Load selected horizon's models
# ------------------------------------------------------------

try:
    ensure_models_loaded(horizon_min)
except Exception as exc:
    st.sidebar.error(
        f"Could not load models:\n{exc}"
    )
    st.stop()


# ------------------------------------------------------------
# Model selector
# ------------------------------------------------------------

model_options = {
    m["id"]: m["name"]
    for m in registry.list_models()
    if m["kind"] == "classical_ml"
}


selected_model_id = st.sidebar.selectbox(
    "Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
)


# ------------------------------------------------------------
# Replay speed
# ------------------------------------------------------------

speed_label = st.sidebar.selectbox(
    "Replay Speed",
    options=list(SPEED_OPTIONS.keys()),
    index=2,
)

speed = SPEED_OPTIONS[speed_label]


# ------------------------------------------------------------
# Rolling window
# ------------------------------------------------------------

window_size = st.sidebar.slider(
    "Rolling Buffer",
    min_value=360,
    max_value=600,
    value=400,
    step=20,
    help=(
        "The model uses historical sensor observations "
        "to calculate rolling features. "
        "The feature pipeline determines when enough "
        "history is available."
    ),
)


# ============================================================
# MODE-SPECIFIC CONTROLS
# ============================================================

uploaded_file = None

if mode == "Demo Mode":

    st.sidebar.subheader("Demo Range")

    demo_start = st.sidebar.number_input(
        "Start row",
        min_value=0,
        max_value=1_516_947,
        value=0,
        step=100,
    )

    demo_length = st.sidebar.number_input(
        "Rows to replay",
        min_value=500,
        max_value=20_000,
        value=3_000,
        step=500,
        help=(
            "Use a manageable number of rows for "
            "a live presentation."
        ),
    )

    demo_end = min(
        demo_start + demo_length,
        1_516_948,
    )

elif mode == "Upload & Test":

    st.sidebar.subheader("Upload Sensor Data")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a raw sensor CSV",
        type=["csv"],
        help=(
            "Upload a raw sensor CSV containing the "
            "timestamp and sensor columns required "
            "by the trained feature pipeline."
        ),
    )

    demo_start = 0
    demo_end = None

else:

    demo_start = 0
    demo_end = None


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.sidebar.columns(2)

start_clicked = col1.button(
    "▶ Start",
    width="stretch",
)

stop_clicked = col2.button(
    "⏸ Stop",
    width="stretch",
)

reset_clicked = st.sidebar.button(
    "↻ Reset",
    width="stretch",
)


# ============================================================
# RESET
# ============================================================

if reset_clicked:

    st.session_state.session = SessionState(
        window_size=window_size
    )

    st.session_state.rows = None
    st.session_state.row_index = 0
    st.session_state.replay_start = 0
    st.session_state.replay_end = None
    st.session_state.is_replaying = False
    st.session_state.event_log = []
    st.session_state.active_model_id = None
    st.session_state.uploaded_filename = None

    st.rerun()


# ============================================================
# STOP
# ============================================================

if stop_clicked:

    st.session_state.is_replaying = False


# ============================================================
# START
# ============================================================

if start_clicked:

    try:

        # ----------------------------------------------------
        # Full Dataset
        # ----------------------------------------------------

        if mode == "Full Dataset":

            if not DATASET_PATH.exists():
                st.error(
                    "MetroPT3 dataset was not found."
                )
                st.stop()

            df = load_dataset(
                str(DATASET_PATH)
            )

            rows = df.to_dict(
                orient="records"
            )

            start_index = 0
            end_index = len(rows)

            st.session_state.uploaded_filename = None


        # ----------------------------------------------------
        # Demo Mode
        # ----------------------------------------------------

        elif mode == "Demo Mode":

            if not DATASET_PATH.exists():
                st.error(
                    "MetroPT3 dataset was not found."
                )
                st.stop()

            df = load_dataset(
                str(DATASET_PATH)
            )

            rows = df.to_dict(
                orient="records"
            )

            start_index = int(demo_start)
            end_index = min(
                int(demo_end),
                len(rows),
            )

            st.session_state.uploaded_filename = None


        # ----------------------------------------------------
        # Upload & Test
        # ----------------------------------------------------

        else:

            if uploaded_file is None:
                st.error(
                    "Please upload a CSV file first."
                )
                st.stop()

            df = load_uploaded_dataset(
                uploaded_file
            )

            rows = df.to_dict(
                orient="records"
            )

            start_index = 0
            end_index = len(rows)

            st.session_state.uploaded_filename = (
                uploaded_file.name
            )


        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if len(rows) == 0:
            st.error(
                "The selected dataset contains no rows."
            )
            st.stop()

        if end_index <= start_index:
            st.error(
                "The selected replay range is empty."
            )
            st.stop()


        # ----------------------------------------------------
        # New session
        # ----------------------------------------------------

        st.session_state.rows = rows

        st.session_state.row_index = start_index

        st.session_state.replay_start = start_index

        st.session_state.replay_end = end_index

        st.session_state.event_log = []

        st.session_state.active_model_id = (
            selected_model_id
        )

        st.session_state.session = SessionState(
            window_size=window_size
        )

        session = st.session_state.session

        session.total_rows = (
            end_index - start_index
        )

        session.selected_model = (
            selected_model_id
        )

        st.session_state.is_replaying = True

        st.rerun()

    except Exception as exc:

        st.error(
            f"Could not start replay:\n{exc}"
        )


# ============================================================
# TITLE
# ============================================================

st.title(
    "Pump Predictive Maintenance — Live Replay"
)

if mode == "Full Dataset":

    st.caption(
        "Real MetroPT3 sensor data replayed row-by-row "
        "through the trained anomaly-detection model."
    )

elif mode == "Demo Mode":

    st.caption(
        "A selected section of real MetroPT3 sensor data "
        "is replayed as a real-time predictive-maintenance "
        "demonstration."
    )

else:

    st.caption(
        "Uploaded raw sensor data is replayed through "
        "the same trained feature-engineering and "
        "prediction pipeline."
    )


# ============================================================
# INFORMATION PANEL
# ============================================================

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:

    st.metric(
        "Prediction Horizon",
        f"{horizon_min} min",
    )

with info_col2:

    st.metric(
        "Model",
        model_options.get(
            selected_model_id,
            selected_model_id,
        ),
    )

with info_col3:

    if mode == "Full Dataset":

        st.metric(
            "Mode",
            "Full Dataset",
        )

    elif mode == "Demo Mode":

        st.metric(
            "Mode",
            "Demo",
        )

    else:

        st.metric(
            "Mode",
            "Upload",
        )

with info_col4:

    st.metric(
        "Replay Speed",
        speed_label,
    )


# ============================================================
# MAIN DISPLAY PLACEHOLDERS
# ============================================================

status_col, alert_col = st.columns(
    [2, 1]
)

sensor_placeholder = st.empty()

event_log_placeholder = st.empty()

chart_placeholder = st.empty()

notification_placeholder = st.empty()


# ============================================================
# EVENT LOG
# ============================================================

def render_event_log():

    with event_log_placeholder:

        st.subheader(
            "🔔 Alert Event Log"
        )

        if not st.session_state.event_log:

            st.info(
                "No alert-level changes yet."
            )

            return

        log_df = pd.DataFrame(
            st.session_state.event_log
        )

        st.dataframe(
            log_df,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# RENDER CURRENT STATE
# ============================================================

def render(prediction):

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    with status_col:

        rows = st.session_state.rows

        start_index = (
            st.session_state.replay_start
        )

        end_index = (
            st.session_state.replay_end
        )

        if rows and end_index is not None:

            total = max(
                1,
                end_index - start_index,
            )

            completed = max(
                0,
                st.session_state.row_index
                - start_index,
            )

            progress = (
                completed / total
            )

            st.progress(
                min(progress, 1.0)
            )

            st.write(
                f"Observations: "
                f"{completed:,} / {total:,}"
            )


    # --------------------------------------------------------
    # Alert
    # --------------------------------------------------------

    with alert_col:

        level = (
            prediction["alert_level"]
            if prediction
            else "NORMAL"
        )

        color = ALERT_COLORS.get(
            level,
            "#95a5a6",
        )

        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:20px;
                border-radius:10px;
                text-align:center;
            ">
                <h2 style="
                    color:white;
                    margin:0;
                ">
                    {level}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )


        if prediction:

            st.metric(
                "Risk score",
                f"{prediction['risk_score']:.4f}",
            )

            st.caption(
                "Model: "
                + model_options.get(
                    prediction["model_id"],
                    prediction["model_id"],
                )
            )

            st.caption(
                f"Prediction horizon: "
                f"{horizon_min} minutes"
            )

            threshold = registry._models[
                prediction["model_id"]
            ].metrics.get("threshold")

            if threshold is not None:

                st.caption(
                    f"Training threshold: "
                    f"{threshold:.4f}"
                )


    # --------------------------------------------------------
    # Current prediction explanation
    # --------------------------------------------------------

    if prediction:

        notification_placeholder.info(
            "The risk score represents the model's "
            f"estimated probability of abnormal/failure "
            f"conditions within the next {horizon_min} "
            "minutes based on the current sensor history."
        )


    # --------------------------------------------------------
    # Simulated time
    # --------------------------------------------------------

    if (
        st.session_state.rows
        and
        st.session_state.row_index > 0
        and
        st.session_state.row_index - 1
        < len(st.session_state.rows)
    ):

        current_row = st.session_state.rows[
            st.session_state.row_index - 1
        ]

        timestamp = current_row.get(
            "timestamp"
        )

        st.caption(
            f"Simulated time: {timestamp}"
        )


    # --------------------------------------------------------
    # Event log
    # --------------------------------------------------------

    render_event_log()


    # --------------------------------------------------------
    # Risk chart
    # --------------------------------------------------------

    if session.history:

        hist_df = pd.DataFrame(
            session.history[-300:]
        )

        st.subheader(
            "Risk History"
        )

        chart_placeholder.line_chart(
            hist_df.set_index(
                "row_index"
            )["risk_score"],
            width="stretch",
        )


    # --------------------------------------------------------
    # Current sensor row
    # --------------------------------------------------------

    if (
        st.session_state.rows
        and
        0 <= st.session_state.row_index - 1
        < len(st.session_state.rows)
    ):

        current_row = st.session_state.rows[
            st.session_state.row_index - 1
        ]

        with sensor_placeholder:

            st.subheader(
                "Current Sensor Reading"
            )

            sensor_df = pd.DataFrame(
                [current_row]
            )

            st.dataframe(
                sensor_df,
                width="stretch",
            )


# ============================================================
# LIVE REPLAY
# ============================================================

if (
    st.session_state.is_replaying
    and
    st.session_state.rows
):

    rows = st.session_state.rows

    i = st.session_state.row_index

    end_index = (
        st.session_state.replay_end
    )

    # --------------------------------------------------------
    # More rows available
    # --------------------------------------------------------

    if (
        end_index is not None
        and
        i < end_index
    ):

        row = rows[i]


        # ----------------------------------------------------
        # Add current sensor reading
        # ----------------------------------------------------

        session.buffer.push(row)

        st.session_state.row_index += 1


        prediction = None


        # ----------------------------------------------------
        # Predict when feature history is ready
        # ----------------------------------------------------

        try:

            score = registry.predict(
                selected_model_id,
                session.buffer.as_list(),
            )

            # ------------------------------------------------
            # The real feature pipeline returns 0 until enough
            # history exists.
            # ------------------------------------------------

            if session.buffer.is_full():

                previous_level = (
                    session.alert_machine
                    .current_level
                    .label
                )

                level = (
                    session.alert_machine.update(
                        score
                    )
                )


                # --------------------------------------------
                # Log alert changes
                # --------------------------------------------

                if (
                    level.label
                    != previous_level
                ):

                    st.session_state.event_log.append(
                        {
                            "row": i,
                            "timestamp": row.get(
                                "timestamp"
                            ),
                            "from": previous_level,
                            "to": level.label,
                            "risk_score": round(
                                score,
                                4,
                            ),
                        }
                    )


                # --------------------------------------------
                # Save prediction
                # --------------------------------------------

                prediction = {
                    "row_index": i,
                    "timestamp": row.get(
                        "timestamp"
                    ),
                    "risk_score": score,
                    "alert_level": level.label,
                    "model_id": selected_model_id,
                }

                session.latest_prediction = (
                    prediction
                )

                session.history.append(
                    prediction
                )

        except Exception as exc:

            st.error(
                f"Prediction error at row {i}: {exc}"
            )

            st.session_state.is_replaying = False

            st.stop()


        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        render(prediction)


        # ----------------------------------------------------
        # Wait according to replay speed
        # ----------------------------------------------------

        time.sleep(
            1.0 / speed
        )


        # ----------------------------------------------------
        # Next replay tick
        # ----------------------------------------------------

        st.rerun()


    # --------------------------------------------------------
    # Replay finished
    # --------------------------------------------------------

    else:

        st.session_state.is_replaying = False

        st.success(
            "Replay finished."
        )

        render(
            session.latest_prediction
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    if session.latest_prediction:

        render(
            session.latest_prediction
        )

    else:

        if mode == "Full Dataset":

            st.info(
                "Full Dataset mode will replay the complete "
                "MetroPT3 dataset. For a presentation, "
                "Demo Mode is recommended."
            )

        elif mode == "Demo Mode":

            st.info(
                "Select a demo range and click ▶ Start "
                "to replay a manageable section of the "
                "real MetroPT3 dataset."
            )

        else:

            st.info(
                "Upload a compatible raw sensor CSV "
                "and click ▶ Start to test it."
            )