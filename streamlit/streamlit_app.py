from __future__ import annotations

from pathlib import Path
from datetime import datetime
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
# SESSION STATE
# ============================================================

defaults = {
    "models_loaded_horizon": None,

    "session": SessionState(
        window_size=400
    ),

    "rows": None,
    "row_index": 0,
    "replay_start": 0,
    "replay_end": None,

    "is_replaying": False,

    "event_log": [],
    "system_log": [],

    "active_model_id": None,
    "active_horizon_min": None,
    "active_mode": None,
    "active_speed_label": None,
    "active_window_size": 400,

    "mode": "Demo Mode",
    "uploaded_filename": None,

    "active_run": None,
    "run_history": [],
    "run_saved": False,
    "run_counter": 0,

    # Replay timing
    "last_replay_tick": None,
    "replay_row_accumulator": 0.0,

    # Persistent latest display state
    "latest_prediction": None,
    "latest_sensor_row": None,

    # UI timing
    "last_chart_update": 0.0,
    "last_sensor_update": 0.0,
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


session = st.session_state.session


# ============================================================
# LOGGING
# ============================================================

def add_system_log(message: str):
    """
    Add a persistent system message to the dashboard session.
    """

    timestamp = datetime.now().strftime(
        "%H:%M:%S"
    )

    st.session_state.system_log.append(
        {
            "time": timestamp,
            "message": message,
        }
    )

    if len(st.session_state.system_log) > 100:

        st.session_state.system_log = (
            st.session_state.system_log[-100:]
        )


def finish_active_run(status: str):
    """
    Save the current run into run_history exactly once.

    Alert events are copied into the saved run so that each
    historical run keeps its own alert-level transition history.
    """

    active_run = st.session_state.active_run

    if active_run is None:
        return

    if st.session_state.run_saved:
        return

    active_run = dict(
        active_run
    )

    active_run["status"] = status

    active_run["ended"] = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    # --------------------------------------------------------
    # Preserve this run's alert events
    # --------------------------------------------------------

    active_run["alert_events"] = list(
        active_run.get(
            "alert_events",
            []
        )
    )

    try:

        active_run["observations"] = (
            st.session_state.session.buffer.length()
        )

    except Exception:

        active_run["observations"] = (
            st.session_state.row_index
            - st.session_state.replay_start
        )

    try:

        active_run["predictions"] = (
            len(
                st.session_state.session.history
            )
        )

    except Exception:

        active_run["predictions"] = 0

    st.session_state.run_history.insert(
        0,
        active_run,
    )

    if len(
        st.session_state.run_history
    ) > 50:

        st.session_state.run_history = (
            st.session_state.run_history[:50]
        )

    st.session_state.run_saved = True

    st.session_state.active_run = (
        active_run
    )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dataset(path: str):

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:

        raise ValueError(
            "The dataset must contain a 'timestamp' column."
        )

    return df


def load_uploaded_dataset(
    uploaded_file
):

    df = pd.read_csv(
        uploaded_file
    )

    if "timestamp" not in df.columns:

        raise ValueError(
            "Uploaded CSV must contain a 'timestamp' column."
        )

    return df


# ============================================================
# MODEL LOADING
# ============================================================

def ensure_models_loaded(
    horizon_min: int
):

    if (
        st.session_state.models_loaded_horizon
        == horizon_min
    ):

        return

    registry.load_all_real_models(
        MODELS_DIR,
        horizon_min,
    )

    registry.load_all_dl_models(
        MODELS_DIR,
        horizon_min,
    )

    st.session_state.models_loaded_horizon = (
        horizon_min
    )

    add_system_log(
        f"Models loaded successfully for "
        f"{horizon_min}-minute horizon."
    )


# ============================================================
# MODEL HELPERS
# ============================================================

def get_model_threshold(
    model_id: str
):

    try:

        model_info = (
            registry._models.get(
                model_id
            )
        )

        if model_info is None:
            return None

        metrics = getattr(
            model_info,
            "metrics",
            None,
        )

        if isinstance(
            metrics,
            dict,
        ):

            return metrics.get(
                "threshold"
            )

        return None

    except Exception:

        return None


def get_model_display_name(
    model_id: str
):

    try:

        for model in registry.list_models():

            if model["id"] == model_id:

                return model["name"]

    except Exception:

        pass

    return model_id


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Controls"
)


controls_locked = (
    st.session_state.is_replaying
)


# ============================================================
# MODE
# ============================================================

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
    ].index(
        st.session_state.mode
    ),
    disabled=controls_locked,
)

st.session_state.mode = mode


# ============================================================
# PREDICTION HORIZON
# ============================================================

horizon_min = st.sidebar.selectbox(
    "Prediction Horizon",
    options=HORIZONS,
    disabled=controls_locked,
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    ensure_models_loaded(
        horizon_min
    )

except Exception as exc:

    st.sidebar.error(
        f"Could not load models:\n{exc}"
    )

    add_system_log(
        f"Model loading error: {exc}"
    )

    st.stop()


# ============================================================
# MODEL TYPE
# ============================================================

model_type = st.sidebar.radio(
    "Model Type",
    options=[
        "Traditional ML",
        "Deep Learning",
    ],
    disabled=controls_locked,
)


# ============================================================
# MODEL SELECTION
# ============================================================

if model_type == "Traditional ML":

    model_options = {
        m["id"]: m["name"]
        for m in registry.list_models()
        if (
            m["kind"]
            == "classical_ml"
            and
            m.get("horizon_min")
            == horizon_min
        )
    }

else:

    model_options = {
        m["id"]: m["name"]
        for m in registry.list_models()
        if (
            m["kind"]
            == "deep_learning"
            and
            m.get("horizon_min")
            == horizon_min
        )
    }


if not model_options:

    st.sidebar.error(
        "No models are available for this selection."
    )

    st.stop()


selected_model_id = st.sidebar.selectbox(
    "Model",
    options=list(
        model_options.keys()
    ),
    format_func=lambda x:
        model_options[x],
    key=(
        f"model_selector_"
        f"{model_type}_"
        f"{horizon_min}"
    ),
    disabled=controls_locked,
)


# ============================================================
# REPLAY SPEED
# ============================================================

speed_label = st.sidebar.selectbox(
    "Replay Speed",
    options=list(
        SPEED_OPTIONS.keys()
    ),
    index=2,
    disabled=controls_locked,
)

speed = SPEED_OPTIONS[
    speed_label
]


# ============================================================
# BUFFER SIZE
# ============================================================

window_size = st.sidebar.slider(
    "Rolling Buffer",
    min_value=360,
    max_value=600,
    value=400,
    step=20,
    disabled=controls_locked,
    help=(
        "The model requires historical sensor "
        "observations to calculate rolling features."
    ),
)


# ============================================================
# MODE-SPECIFIC CONTROLS
# ============================================================

uploaded_file = None


if mode == "Demo Mode":

    st.sidebar.subheader(
        "Demo Range"
    )

    demo_start = st.sidebar.number_input(
        "Start row",
        min_value=0,
        max_value=1_516_947,
        value=0,
        step=100,
        disabled=controls_locked,
    )

    demo_length = st.sidebar.number_input(
        "Rows to replay",
        min_value=500,
        max_value=20_000,
        value=3_000,
        step=500,
        disabled=controls_locked,
        help=(
            "Use a manageable number of rows "
            "for a live presentation."
        ),
    )

    demo_end = min(
        demo_start + demo_length,
        1_516_948,
    )


elif mode == "Upload & Test":

    st.sidebar.subheader(
        "Upload Sensor Data"
    )

    uploaded_file = st.sidebar.file_uploader(
        "Choose a raw sensor CSV",
        type=["csv"],
        disabled=controls_locked,
        help=(
            "Upload a raw sensor CSV containing "
            "the timestamp and sensor columns "
            "required by the trained feature pipeline."
        ),
    )

    demo_start = 0
    demo_end = None


else:

    demo_start = 0
    demo_end = None


# ============================================================
# CONTROL BUTTONS
# ============================================================

col1, col2 = st.sidebar.columns(
    2
)


start_clicked = col1.button(
    "▶ Start",
    width="stretch",
    disabled=(
        st.session_state.is_replaying
    ),
)


stop_clicked = col2.button(
    "⏸ Stop",
    width="stretch",
    disabled=(
        not st.session_state.is_replaying
    ),
)


reset_clicked = st.sidebar.button(
    "↻ Reset",
    width="stretch",
)


# ============================================================
# RESET
# ============================================================

if reset_clicked:

    if st.session_state.is_replaying:

        finish_active_run(
            "RESET"
        )

        add_system_log(
            "Current replay reset by user."
        )

    else:

        add_system_log(
            "Dashboard reset."
        )


    st.session_state.session = (
        SessionState(
            window_size=window_size
        )
    )

    st.session_state.rows = None

    st.session_state.row_index = 0

    st.session_state.replay_start = 0

    st.session_state.replay_end = None

    st.session_state.is_replaying = False

    st.session_state.event_log = []

    st.session_state.active_model_id = None

    st.session_state.active_horizon_min = None

    st.session_state.active_mode = None

    st.session_state.active_speed_label = None

    st.session_state.active_window_size = (
        window_size
    )

    st.session_state.uploaded_filename = None

    st.session_state.active_run = None

    st.session_state.run_saved = False

    st.session_state.last_replay_tick = None

    st.session_state.replay_row_accumulator = 0.0

    st.session_state.latest_prediction = None

    st.session_state.latest_sensor_row = None

    st.session_state.last_chart_update = 0.0

    st.session_state.last_sensor_update = 0.0

    st.rerun()


# ============================================================
# STOP
# ============================================================

if stop_clicked:

    finish_active_run(
        "STOPPED"
    )

    add_system_log(
        "Replay stopped by user."
    )

    st.session_state.is_replaying = False

    st.session_state.last_replay_tick = None

    st.session_state.replay_row_accumulator = 0.0

    st.session_state.last_chart_update = 0.0

    st.session_state.last_sensor_update = 0.0

    st.rerun()


# ============================================================
# START
# ============================================================

if start_clicked:

    try:

        # ----------------------------------------------------
        # Select dataset
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

            end_index = len(
                rows
            )

            st.session_state.uploaded_filename = (
                None
            )


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

            start_index = int(
                demo_start
            )

            end_index = min(
                int(demo_end),
                len(rows),
            )

            st.session_state.uploaded_filename = (
                None
            )


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

            end_index = len(
                rows
            )

            st.session_state.uploaded_filename = (
                uploaded_file.name
            )


        # ----------------------------------------------------
        # Validate replay range
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
        # New run
        # ----------------------------------------------------

        st.session_state.run_counter += 1

        run_id = (
            st.session_state.run_counter
        )

        model_name = (
            get_model_display_name(
                selected_model_id
            )
        )

        st.session_state.active_run = {

            "run": run_id,

            "started": (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),

            "mode": mode,

            "model": model_name,

            "model_id": selected_model_id,

            "horizon": horizon_min,

            "speed": speed_label,

            "status": "RUNNING",

            "dataset": (
                st.session_state.uploaded_filename
                if st.session_state.uploaded_filename
                else "MetroPT3"
            ),

            # ------------------------------------------------
            # NEW:
            # Every run owns its own alert history.
            # ------------------------------------------------
            "alert_events": [],
        }

        st.session_state.run_saved = False


        # ----------------------------------------------------
        # Reset model session for new run
        # ----------------------------------------------------

        st.session_state.session = (
            SessionState(
                window_size=window_size
            )
        )

        session = (
            st.session_state.session
        )

        session.total_rows = (
            end_index - start_index
        )

        session.selected_model = (
            selected_model_id
        )


        # ----------------------------------------------------
        # Store active settings
        # ----------------------------------------------------

        st.session_state.rows = rows

        st.session_state.row_index = (
            start_index
        )

        st.session_state.replay_start = (
            start_index
        )

        st.session_state.replay_end = (
            end_index
        )

        # ----------------------------------------------------
        # NEW RUN = NEW LIVE EVENT LOG
        # ----------------------------------------------------

        st.session_state.event_log = []

        st.session_state.active_model_id = (
            selected_model_id
        )

        st.session_state.active_horizon_min = (
            horizon_min
        )

        st.session_state.active_mode = (
            mode
        )

        st.session_state.active_speed_label = (
            speed_label
        )

        st.session_state.active_window_size = (
            window_size
        )

        # ----------------------------------------------------
        # Reset replay/UI timers
        # ----------------------------------------------------

        st.session_state.last_replay_tick = (
            time.monotonic()
        )

        st.session_state.replay_row_accumulator = 0.0

        st.session_state.latest_prediction = None

        st.session_state.latest_sensor_row = None

        st.session_state.last_chart_update = 0.0

        st.session_state.last_sensor_update = 0.0

        st.session_state.is_replaying = True


        # ----------------------------------------------------
        # System log
        # ----------------------------------------------------

        add_system_log(
            f"Run {run_id} started: "
            f"{model_name}, "
            f"{horizon_min} min, "
            f"{speed_label}, "
            f"{mode}."
        )


        # ----------------------------------------------------
        # Full page rerun ONLY when Start is pressed
        # ----------------------------------------------------

        st.rerun()


    except Exception as exc:

        st.error(
            f"Could not start replay:\n{exc}"
        )

        add_system_log(
            f"Start error: {exc}"
        )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "Pump Predictive Maintenance — Live Replay"
)

st.caption(
    "Historical sensor data is replayed sequentially "
    "to simulate a real-time predictive-maintenance system."
)


# ============================================================
# EXPLANATION PANEL
# ============================================================

with st.expander(
    "How this dashboard works",
    expanded=False,
):

    st.markdown(
        """
### What you are seeing

The dashboard replays historical pump sensor observations
sequentially to simulate a real-time monitoring system.

For each observation:

1. The sensor reading enters the rolling buffer.
2. The trained feature pipeline processes the available history.
3. The selected model estimates the current risk.
4. The risk is converted into a dashboard alert level.
5. The result is added to the current run.

### Risk score

The risk score ranges from **0 to 1**.

A higher value means the trained model estimates a higher
likelihood of the target abnormal/failure condition within
the selected prediction horizon.

The risk score is **not a measurement of physical damage**.

### Alert levels

- **NORMAL** — low model risk
- **WATCH** — elevated risk
- **WARNING** — high model risk
- **ALERT** — very high model risk

### Historical replay

This is a simulation using historical data rather than a
live sensor connection.
"""
    )


# ============================================================
# STATIC CURRENT RUN INFORMATION
# ============================================================

if (
    st.session_state.active_run
    is not None
):

    active_run = (
        st.session_state.active_run
    )

    st.markdown("### Live Monitoring Session")

    run_col1, run_col2, run_col3, run_col4 = st.columns(4)

    with run_col1:
        st.metric(
            "Selected Model",
            active_run["model"],
            help="The trained model currently being used for failure-risk prediction.",
        )

    with run_col2:
        st.metric(
            "Prediction Horizon",
            f"{active_run['horizon']} min",
            help="How far ahead the model is attempting to predict a failure.",
        )

    with run_col3:
        st.metric(
            "Replay Speed",
            active_run["speed"],
            help="Controls how quickly the historical sensor data is replayed.",
        )

    with run_col4:
        run_status = active_run["status"]

        if run_status == "RUNNING":
            status_icon = "🟢"
        elif run_status == "COMPLETED":
            status_icon = "✅"
        elif run_status == "STOPPED":
            status_icon = "⏸️"
        elif run_status == "ERROR":
            status_icon = "🔴"
        else:
            status_icon = "⚪"

        st.metric(
            "Session Status",
            f"{status_icon} {run_status}",
        )

    st.caption(
        f"Run #{active_run['run']} • "
        f"{active_run['mode']} • "
        f"Dataset: {active_run['dataset']}"
    )

# ============================================================
# LIVE REPLAY FRAGMENT
# ============================================================

@st.fragment(run_every=0.1)
def replay_tick():

    # ========================================================
    # IF NOT REPLAYING
    # ========================================================

    if not st.session_state.is_replaying:

        return


    # ========================================================
    # REPLAY TIMING
    # ========================================================

    now = time.monotonic()

    active_speed_label = (
        st.session_state.active_speed_label
    )

    if active_speed_label not in SPEED_OPTIONS:

        active_speed_label = "10×"

    speed = SPEED_OPTIONS[
        active_speed_label
    ]

    last_tick = (
        st.session_state.last_replay_tick
    )

    if last_tick is None:

        st.session_state.last_replay_tick = now

        return


    elapsed = (
        now - last_tick
    )

    st.session_state.replay_row_accumulator += (
        elapsed * speed
    )

    rows_to_process = int(
        st.session_state.replay_row_accumulator
    )

    st.session_state.replay_row_accumulator -= (
        rows_to_process
    )

    st.session_state.last_replay_tick = now


    # ========================================================
    # DATASET
    # ========================================================

    rows = (
        st.session_state.rows
    )

    if not rows:

        st.session_state.is_replaying = False

        finish_active_run(
            "ERROR"
        )

        add_system_log(
            "Replay stopped because no dataset rows were available."
        )

        st.rerun()


    i = (
        st.session_state.row_index
    )

    end_index = (
        st.session_state.replay_end
    )


    # ========================================================
    # REPLAY FINISHED BEFORE PROCESSING
    # ========================================================

    if (
        end_index is None
        or
        i >= end_index
    ):

        run_number = None

        if (
            st.session_state.active_run
            is not None
        ):

            run_number = (
                st.session_state.active_run[
                    "run"
                ]
            )

        finish_active_run(
            "COMPLETED"
        )

        if run_number is not None:

            add_system_log(
                f"Run {run_number} completed."
            )

        st.session_state.is_replaying = False

        st.session_state.last_replay_tick = None

        st.session_state.replay_row_accumulator = 0.0

        st.session_state.last_chart_update = 0.0

        st.session_state.last_sensor_update = 0.0

        st.rerun()


    # ========================================================
    # PROCESS REPLAY OBSERVATIONS
    # ========================================================

    session = (
        st.session_state.session
    )

    rows_to_process = min(
        rows_to_process,
        end_index - i,
    )

    latest_row = (
        st.session_state.latest_sensor_row
    )


    for _ in range(
        rows_to_process
    ):

        i = (
            st.session_state.row_index
        )

        if i >= end_index:

            break


        row = rows[i]

        latest_row = row

        st.session_state.latest_sensor_row = (
            row
        )


        # ----------------------------------------------------
        # Add observation to rolling buffer
        # ----------------------------------------------------

        session.buffer.push(
            row
        )

        st.session_state.row_index += 1


        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        if not session.buffer.is_full():

            continue


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        try:

            active_model_id = (
                st.session_state.active_model_id
            )

            if active_model_id is None:

                raise ValueError(
                    "No active model is attached "
                    "to the current run."
                )


            score = registry.predict(
                active_model_id,
                session.buffer.as_list(),
            )


            # ------------------------------------------------
            # Alert state
            # ------------------------------------------------

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

            current_level = (
                level.label
            )


            # ------------------------------------------------
            # Alert event
            # ------------------------------------------------

            if (
                current_level
                != previous_level
            ):

                event = {
                    "row": i,

                    "timestamp": row.get(
                        "timestamp"
                    ),

                    "from": previous_level,

                    "to": current_level,

                    "risk_score": round(
                        float(score),
                        4,
                    ),
                }

                # --------------------------------------------
                # Live current-run event log
                # --------------------------------------------

                st.session_state.event_log.append(
                    event
                )

                # --------------------------------------------
                # NEW:
                # Store the same event inside this run.
                # --------------------------------------------

                if (
                    st.session_state.active_run
                    is not None
                ):

                    st.session_state.active_run.setdefault(
                        "alert_events",
                        []
                    ).append(
                        event
                    )


            # ------------------------------------------------
            # Prediction object
            # ------------------------------------------------

            latest_prediction = {

                "row_index": i,

                "timestamp": row.get(
                    "timestamp"
                ),

                "risk_score": float(
                    score
                ),

                "alert_level": (
                    current_level
                ),

                "model_id": (
                    active_model_id
                ),
            }


            session.latest_prediction = (
                latest_prediction
            )

            session.history.append(
                latest_prediction
            )

            st.session_state.latest_prediction = (
                latest_prediction
            )


        except Exception as exc:

            st.error(
                f"Prediction error at row {i}: {exc}"
            )

            add_system_log(
                f"Prediction error at row {i}: {exc}"
            )

            finish_active_run(
                "ERROR"
            )

            st.session_state.is_replaying = False

            st.session_state.last_replay_tick = None

            st.session_state.replay_row_accumulator = 0.0

            st.rerun()


    # ========================================================
    # CALCULATE PROGRESS
    # ========================================================

    processed = (
        st.session_state.row_index
        -
        st.session_state.replay_start
    )

    total = (
        st.session_state.replay_end
        -
        st.session_state.replay_start
    )

    progress = (
        processed / total
        if total > 0
        else 0
    )

    progress = min(
        max(
            progress,
            0,
        ),
        1,
    )


    # ========================================================
    # REPLAY COMPLETION CHECK
    # ========================================================

    if (
        st.session_state.row_index
        >= end_index
    ):

        run_number = None

        if (
            st.session_state.active_run
            is not None
        ):

            run_number = (
                st.session_state.active_run[
                    "run"
                ]
            )

        finish_active_run(
            "COMPLETED"
        )

        if run_number is not None:

            add_system_log(
                f"Run {run_number} completed."
            )

        st.session_state.is_replaying = False

        st.session_state.last_replay_tick = None

        st.session_state.replay_row_accumulator = 0.0

        st.session_state.last_chart_update = 0.0

        st.session_state.last_sensor_update = 0.0

        st.rerun()


    # ========================================================
    # WARM-UP DISPLAY
    # ========================================================

    if not session.buffer.is_full():

        status_col, alert_col = (
            st.columns(
                [2, 1]
            )
        )

        with status_col:

            st.subheader(
                "Replay Status"
            )

            st.progress(
                progress
            )

            st.info(
                "Preparing monitoring — "
                "collecting initial sensor history."
            )

            st.caption(
                f"{processed:,} / "
                f"{total:,} observations processed. "
                "Predictions begin when enough history "
                "is available."
            )


        with alert_col:

            st.markdown(
                """
                <div style="
                    padding: 18px;
                    border-radius: 10px;
                    text-align: center;
                    background-color: #808080;
                    color: white;
                    font-size: 22px;
                    font-weight: bold;
                ">
                    PREPARING
                </div>
                """,
                unsafe_allow_html=True,
            )

        return


    # ========================================================
    # USE PERSISTED LATEST PREDICTION
    # ========================================================

    prediction = (
        st.session_state.latest_prediction
    )

    if prediction is None:

        return


    # ========================================================
    # UI TIMING
    # ========================================================

    current_time = time.monotonic()

    chart_should_update = (
        current_time
        -
        st.session_state.last_chart_update
        >= 0.3
    )

    sensor_should_update = (
        current_time
        -
        st.session_state.last_sensor_update
        >= 0.5
    )

    if chart_should_update:

        st.session_state.last_chart_update = (
            current_time
        )

    if sensor_should_update:

        st.session_state.last_sensor_update = (
            current_time
        )


    # ========================================================
    # LIVE STATUS
    # ========================================================

    status_col, alert_col = (
        st.columns(
            [2, 1]
        )
    )


    with status_col:

        st.subheader(
            "Replay Status"
        )

        st.progress(
            progress
        )

        st.caption(
            f"{processed:,} / "
            f"{total:,} observations processed"
        )


    with alert_col:

        alert_level = (
            prediction[
                "alert_level"
            ]
        )

        alert_color = (
            ALERT_COLORS.get(
                alert_level,
                "#808080",
            )
        )

        st.markdown(
            f"""
            <div style="
                padding: 18px;
                border-radius: 10px;
                text-align: center;
                background-color: {alert_color};
                color: white;
                font-size: 26px;
                font-weight: bold;
            ">
                {alert_level}
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # CURRENT PREDICTION
    # ========================================================

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    metric1.metric(
        "Risk Score",
        f"{prediction['risk_score']:.3f}",
    )


    metric2.metric(
        "Prediction Horizon",
        f"{st.session_state.active_horizon_min} min",
    )


    metric3.metric(
        "Model",
        get_model_display_name(
            prediction["model_id"]
        ),
    )


    threshold = (
        get_model_threshold(
            prediction["model_id"]
        )
    )


    if threshold is not None:

        metric4.metric(
            "Model Threshold",
            f"{float(threshold):.3f}",
        )

    else:

        metric4.metric(
            "Model Threshold",
            "N/A",
        )


    # ========================================================
    # RISK EXPLANATION
    # ========================================================

    st.info(
        f"""
**Current model estimate**

The model currently estimates a risk score of
**{prediction['risk_score']:.3f}**.

For the selected **{st.session_state.active_horizon_min}-minute
horizon**, this represents the model's estimated likelihood
of the target abnormal/failure condition within that horizon.

This score is a model output, not a direct measurement of
physical damage.
"""
    )


    # ========================================================
    # SIMULATED TIME
    # ========================================================

    st.write(
        f"**Simulated sensor time:** "
        f"{prediction['timestamp']}"
    )


    # ========================================================
    # RISK HISTORY
    # ========================================================

    st.subheader(
        "Pump Condition Trend"
    )

    history = pd.DataFrame(
        session.history
    )

    if not history.empty:

        chart_data = (
            history[
                [
                    "row_index",
                    "risk_score",
                ]
            ]
            .copy()
            .tail(150)
            .set_index(
                "row_index"
            )
        )

        st.line_chart(
            chart_data,
            width="stretch",
        )

        st.caption(
            "Recent model-estimated condition trend. "
            "Higher values indicate higher estimated risk."
        )


    # ========================================================
    # CURRENT SENSOR READING
    # ========================================================

    if latest_row is not None:

        st.subheader(
            "Current Sensor Reading"
        )

        st.caption(
            "Latest historical sensor observation "
            "being replayed."
        )

        current_row_df = pd.DataFrame(
            [latest_row]
        )

        st.dataframe(
            current_row_df,
            width="stretch",
            hide_index=True,
        )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "Technical prediction details"
    ):

        st.write(
            {
                "Model ID": prediction[
                    "model_id"
                ],

                "Model name": (
                    get_model_display_name(
                        prediction[
                            "model_id"
                        ]
                    )
                ),

                "Prediction horizon": (
                    f"{st.session_state.active_horizon_min} minutes"
                ),

                "Risk score": round(
                    prediction[
                        "risk_score"
                    ],
                    6,
                ),

                "Alert level": prediction[
                    "alert_level"
                ],

                "Replay row": prediction[
                    "row_index"
                ],

                "Timestamp": prediction[
                    "timestamp"
                ],
            }
        )


    # ========================================================
    # CURRENT RUN ALERT HISTORY
    # ========================================================

    current_run = (
        st.session_state.active_run
    )

    if current_run is not None:

        st.subheader(
            f"Current Run #{current_run['run']} — Alert History"
        )

        if st.session_state.event_log:

            st.caption(
                "Alert-level transitions recorded during this run."
            )

            event_df = pd.DataFrame(
                st.session_state.event_log
            )

            st.dataframe(
                event_df.tail(20),
                width="stretch",
                hide_index=True,
            )

        else:

            st.info(
                "No alert-level changes recorded "
                "during this run yet."
            )


# ============================================================
# RUN THE REPLAY FRAGMENT
# ============================================================

if st.session_state.is_replaying:

    replay_tick()


# ============================================================
# POST-REPLAY CURRENT RUN ALERT HISTORY
#
# This remains visible after the fragment stops.
# ============================================================

if (
    not st.session_state.is_replaying
    and
    st.session_state.active_run
    is not None
):

    current_run = (
        st.session_state.active_run
    )

    current_events = (
        current_run.get(
            "alert_events",
            []
        )
    )

    st.subheader(
        f"Current Run #{current_run['run']} — Alert History"
    )

    if current_events:

        st.caption(
            "Alert-level transitions recorded during this run."
        )

        current_event_df = pd.DataFrame(
            current_events
        )

        st.dataframe(
            current_event_df.tail(20),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No alert-level changes occurred during this run."
        )


# ============================================================
# NON-REPLAY INFORMATION
# ============================================================

if not st.session_state.is_replaying:

    if (
        st.session_state.active_run
        is None
        and
        st.session_state.rows
        is None
    ):

        st.info(
            "Choose a dataset mode, prediction horizon "
            "and trained model from the sidebar, then "
            "press **Start** to begin the historical replay."
        )


# ============================================================
# PREVIOUS RUNS
# ============================================================

if st.session_state.run_history:

    st.subheader(
        "Previous Runs"
    )

    history_display = []

    for run in (
        st.session_state.run_history
    ):

        alert_events = run.get(
            "alert_events",
            []
        )

        change_count = len(
            alert_events
        )

        if change_count == 0:

            status_changes = (
                "No changes"
            )

        elif change_count == 1:

            status_changes = (
                "1 change"
            )

        else:

            status_changes = (
                f"{change_count} changes"
            )


        history_display.append(
            {
                "Run": run.get(
                    "run"
                ),

                "Started": run.get(
                    "started"
                ),

                "Mode": run.get(
                    "mode"
                ),

                "Model": run.get(
                    "model"
                ),

                "Horizon": (
                    f"{run.get('horizon')} min"
                    if run.get(
                        "horizon"
                    )
                    is not None
                    else "-"
                ),

                "Speed": run.get(
                    "speed"
                ),

                "Status": run.get(
                    "status"
                ),

                "Status Changes": (
                    status_changes
                ),

                "Observations": run.get(
                    "observations",
                    0,
                ),

                "Predictions": run.get(
                    "predictions",
                    0,
                ),
            }
        )


    # --------------------------------------------------------
    # Previous Runs Summary Table
    # --------------------------------------------------------

    st.dataframe(
        pd.DataFrame(
            history_display
        ),
        width="stretch",
        hide_index=True,
    )


    # --------------------------------------------------------
    # Run Detail Selector
    # --------------------------------------------------------

    st.subheader(
        "View Run Details"
    )

    run_options = {}

    for run in (
        st.session_state.run_history
    ):

        alert_events = run.get(
            "alert_events",
            []
        )

        change_count = len(
            alert_events
        )

        if change_count == 0:

            change_text = (
                "No changes"
            )

        elif change_count == 1:

            change_text = (
                "1 change"
            )

        else:

            change_text = (
                f"{change_count} changes"
            )

        label = (
            f"Run #{run.get('run')} "
            f"— {change_text} "
            f"— {run.get('status')}"
        )

        run_options[label] = run


    selected_run_label = st.selectbox(
        "Select a previous run",
        options=list(
            run_options.keys()
        ),
    )

    selected_run = run_options[
        selected_run_label
    ]


    # --------------------------------------------------------
    # Selected Run Status Changes
    # --------------------------------------------------------

    selected_events = (
        selected_run.get(
            "alert_events",
            []
        )
    )

    with st.expander(
        f"Status Change Details — "
        f"Run #{selected_run.get('run')}",
        expanded=True,
    ):

        if selected_events:

            st.caption(
                "Alert-level transitions recorded during this run."
            )

            selected_event_df = pd.DataFrame(
                selected_events
            )

            st.dataframe(
                selected_event_df,
                width="stretch",
                hide_index=True,
            )

        else:

            st.info(
                "No alert-level changes occurred during this run."
            )


# ============================================================
# SYSTEM LOG — SIDEBAR DIAGNOSTICS
# ============================================================

with st.sidebar.expander(
    "Diagnostics / System Log",
    expanded=False,
):

    if st.session_state.system_log:

        system_df = pd.DataFrame(
            st.session_state.system_log
        )

        st.dataframe(
            system_df.tail(50),
            width="stretch",
            hide_index=True,
        )

    else:

        st.caption(
            "No system messages recorded."
        )