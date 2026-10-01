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


# ============================================================
# PROJECT PATHS
# ============================================================

# Current file:
# project/streamlit/streamlit_app.py
#
# parents[0] = streamlit
# parents[1] = project root

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
# SETTINGS
# ============================================================

# Start with the 30-minute models.
#
# You have:
# 20 min
# 30 min
# 40 min
# 50 min
#
# We can add a selector later.
ACTIVE_HORIZON_MIN = 30


# ============================================================
# LOAD REAL MODELS
# ============================================================

if "models_loaded" not in st.session_state:

    registry.load_all_real_models(
        models_dir=str(MODELS_DIR),
        horizon_min=ACTIVE_HORIZON_MIN,
    )

    st.session_state.models_loaded = True


# ============================================================
# SESSION STATE
# ============================================================

if "session" not in st.session_state:

    st.session_state.session = SessionState(
        window_size=400
    )


if "rows" not in st.session_state:

    st.session_state.rows = None


if "row_index" not in st.session_state:

    st.session_state.row_index = 0


if "is_replaying" not in st.session_state:

    st.session_state.is_replaying = False


if "event_log" not in st.session_state:

    st.session_state.event_log = []


if "active_model_id" not in st.session_state:

    st.session_state.active_model_id = None


session = st.session_state.session


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset(path: str):

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:

        raise ValueError(
            "The dataset must contain a timestamp column."
        )

    return df


# ============================================================
# ALERT DISPLAY
# ============================================================

ALERT_COLORS = {
    "NORMAL": "#2ecc71",
    "WATCH": "#f1c40f",
    "WARNING": "#e67e22",
    "ALERT": "#e74c3c",
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Controls")


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

st.sidebar.write("Dataset")

if DATASET_PATH.exists():

    st.sidebar.success(
        "Real MetroPT3 dataset found."
    )

else:

    st.sidebar.error(
        f"Dataset not found:\n{DATASET_PATH}"
    )


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model_options = {
    m["id"]: m["name"]
    for m in registry.list_models()
}


selected_model_id = st.sidebar.selectbox(
    "Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
)


# ------------------------------------------------------------
# Rolling window
# ------------------------------------------------------------

window_size = st.sidebar.slider(
    "Rolling window size",
    min_value=360,
    max_value=600,
    value=400,
    step=20,
    help=(
        "The trained features use rolling windows up to "
        "60 minutes. At approximately 10-second sampling, "
        "360 rows cover 60 minutes."
    ),
)


# ------------------------------------------------------------
# Replay speed
# ------------------------------------------------------------

speed = st.sidebar.slider(
    "Replay speed (rows/sec)",
    min_value=1,
    max_value=100,
    value=20,
)


# ------------------------------------------------------------
# Buttons
# ------------------------------------------------------------

col1, col2 = st.sidebar.columns(2)

start_clicked = col1.button("▶ Start")

stop_clicked = col2.button("⏸ Stop")

reset_clicked = st.sidebar.button("⟲ Reset")


# ============================================================
# RESET
# ============================================================

if reset_clicked:

    st.session_state.session = SessionState(
        window_size=window_size
    )

    st.session_state.rows = None
    st.session_state.row_index = 0
    st.session_state.is_replaying = False
    st.session_state.event_log = []
    st.session_state.active_model_id = None

    st.rerun()


# ============================================================
# START
# ============================================================

if start_clicked:

    if not DATASET_PATH.exists():

        st.error(
            "MetroPT3 dataset was not found."
        )

        st.stop()


    # Load the real CSV.
    df = load_dataset(
        str(DATASET_PATH)
    )

    # Convert to raw sensor dictionaries.
    rows = df.to_dict(
        orient="records"
    )

    st.session_state.rows = rows

    st.session_state.row_index = 0

    st.session_state.event_log = []

    st.session_state.active_model_id = (
        selected_model_id
    )

    # Create a fresh session with selected window.
    st.session_state.session = SessionState(
        window_size=window_size
    )

    session = st.session_state.session

    session.total_rows = len(rows)

    session.selected_model = selected_model_id

    st.session_state.is_replaying = True


# ============================================================
# STOP
# ============================================================

if stop_clicked:

    st.session_state.is_replaying = False


# ============================================================
# TITLE
# ============================================================

st.title(
    "Pump Predictive Maintenance — Live Replay"
)

st.caption(
    "Real MetroPT3 sensor data replayed row-by-row "
    "through the trained anomaly-detection model."
)


# ============================================================
# MAIN DISPLAY
# ============================================================

status_col, alert_col = st.columns([2, 1])

sensor_placeholder = st.empty()

event_log_placeholder = st.empty()

chart_placeholder = st.empty()

notification_placeholder = st.empty()


# ============================================================
# EVENT LOG
# ============================================================

def render_event_log():

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
        use_container_width=True,
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

        if st.session_state.rows:

            progress = (
                st.session_state.row_index
                / max(
                    1,
                    len(st.session_state.rows)
                )
            )

            st.progress(
                min(progress, 1.0)
            )

            st.write(
                f"Row "
                f"{st.session_state.row_index:,}"
                f" / "
                f"{len(st.session_state.rows):,}"
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
            "#95a5a6"
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
                    prediction['model_id'],
                    prediction['model_id']
                )
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
    # Notification
    # --------------------------------------------------------

    if prediction:

        if level in (
            "WARNING",
            "ALERT",
        ):

            notification_placeholder.warning(
                f"⚠️ {level}: "
                "The trained model is detecting an "
                "increasing abnormal-risk pattern."
            )

        else:

            notification_placeholder.empty()


    # --------------------------------------------------------
    # Event log
    # --------------------------------------------------------

    render_event_log()


    # --------------------------------------------------------
    # Risk chart
    # --------------------------------------------------------

    if session.history:

        hist_df = pd.DataFrame(
            session.history[-200:]
        )

        chart_placeholder.line_chart(
            hist_df.set_index(
                "row_index"
            )["risk_score"]
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

        sensor_placeholder.dataframe(
            pd.DataFrame(
                [current_row]
            ),
            use_container_width=True,
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


    # --------------------------------------------------------
    # More rows available
    # --------------------------------------------------------

    if i < len(rows):

        row = rows[i]


        # ----------------------------------------------------
        # Add current sensor reading to rolling buffer
        # ----------------------------------------------------

        session.buffer.push(row)

        st.session_state.row_index += 1


        prediction = None


        # ----------------------------------------------------
        # Only predict once enough history exists
        # ----------------------------------------------------

        if session.buffer.is_full():

            score = registry.predict(
                selected_model_id,
                session.buffer.as_list(),
            )


            # -----------------------------------------------
            # Alert state
            # -----------------------------------------------

            previous_level = (
                session.alert_machine
                .current_level
                .label
            )

            level = (
                session.alert_machine
                .update(score)
            )


            # -----------------------------------------------
            # Log alert changes
            # -----------------------------------------------

            if level.label != previous_level:

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
                            4
                        ),
                    }
                )


            # -----------------------------------------------
            # Save prediction
            # -----------------------------------------------

            prediction = {
                "row_index": i,
                "timestamp": row.get(
                    "timestamp"
                ),
                "risk_score": score,
                "alert_level": level.label,
                "model_id": selected_model_id,
            }


            session.latest_prediction = prediction

            session.history.append(
                prediction
            )


        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        render(prediction)


        # ----------------------------------------------------
        # Wait before next sensor reading
        # ----------------------------------------------------

        time.sleep(
            1.0 / speed
        )


        # ----------------------------------------------------
        # Run next replay tick
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

        st.info(
            "Select a model and click ▶ Start "
            "to begin the real MetroPT3 replay."
        )