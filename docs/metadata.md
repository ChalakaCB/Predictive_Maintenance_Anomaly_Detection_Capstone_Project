# MetroPT-3 feature and column guide

This document describes the schema produced by `src/preprocess.py`. It is the
human-readable reference for the final CSV files; no separate machine-readable
metadata file is required. The same constants are defined in
`src/preprocess.py` so that the feature names and target definitions stay
reproducible.

## 1. What one output row represents

Each row represents one timestamp from the original MetroPT-3 stream. The
feature values are calculated using that row and observations at earlier
timestamps only. The output files are:

```text
train.csv
val.csv
test.csv
```

The filename identifies the chronological split, so a `split` column is not
written to the final files. The current schema contains 181 columns:

| Group | Count | Purpose |
|---|---:|---|
| Row and audit context | 3 | Timestamp, documented failure-window status, and event identifier |
| Warning targets | 4 | One target for each 5-, 10-, 20-, and 30-minute horizon |
| Data-quality gate | 1 | Whether enough 60-minute history is available |
| Model features | 173 | Current readings, sampling quality, and causal rolling features |

## 2. Original sensor columns (grouped overview)

The raw dataset contains two practical sensor families. The raw names are kept
inside the generated feature names, but are grouped here rather than repeated
one by one.

### Analogue process measurements

These are continuous physical measurements:

- pressure-related readings (`TP2`, `TP3`, `H1`, `DV_pressure`);
- reservoir/level measurement (`Reservoirs`);
- temperature (`Oil_temperature`);
- electrical load (`Motor_current`).

They are suitable for level, variability, range, and trend-like context. For
example, a rising `TP2` rolling mean or a large `Oil_temperature` rolling
standard deviation can describe a change in operating behaviour.

### Digital operating states

These are discrete operating or switch signals, normally represented as 0/1:
compressor state, electric valve, cooling towers, motor protection/impulse
signals, pressure switch, oil-level switch, and flow impulses. Their rolling
features describe the fraction of time a state is active and how often it
changes, rather than treating the state as a continuous physical quantity.

## 3. Audit and target columns (do not use as model inputs)

These columns are intentionally exported so that a result can be inspected and
evaluated, but they describe the label or the row itself rather than the
available sensor evidence.

### `timestamp`

The observation time. It is needed for chronological ordering and event-level
evaluation, but should not be passed directly to a model as a numeric feature.

### `in_failure_window`

`True` when the timestamp lies inside one of the four documented failure
intervals (F1-F4). Rows in these intervals are not warning examples. The
intervals represent the reported maintenance/failure periods, not a sensor
prediction made by the pipeline.

### `failure_id`

The event identifier (`F1`, `F2`, `F3`, or `F4`) for rows inside a documented
failure interval; otherwise it is empty. It is useful for grouping metrics by
event and must be excluded from `X`.

### `horizon_label_<N>min`

There is one binary target for each `N` in `{5, 10, 20, 30}`. A value is `1`
when the next documented failure starts more than 0 and no more than `N`
minutes after the current timestamp, and the current row is outside a failure
window. Otherwise it is `0`.

For example, if F4 starts at 14:30, a row at 14:12 has only the 20- and
30-minute labels equal to `1` (it is 18 minutes before the event), while a row
at 14:26 has all four labels equal to `1` (it is 4 minutes before the event).
A row inside F4 has all warning labels equal to `0`.

The internal countdown used to construct these labels is not exported, so the
model cannot receive a direct “minutes to failure” answer.

### `history_ready_60min`

This is a quality-control flag, not a sensor feature. It is `1` when at least
30 observations are available in the causal 60-minute context and `0` during
the initial warm-up or after an unusually long data gap. Rows with `0` can be
filtered before modelling; keeping the flag in the CSV makes that decision
auditable.

## 4. Current-value features

The pattern is:

```text
current_<raw_sensor_name>
```

It is the latest reading available at the row timestamp, with no look-ahead.
Examples:

- `current_TP2` is the current pressure reading at TP2;
- `current_Oil_temperature` is the current oil temperature;
- `current_COMP` is the current compressor state (typically 0 or 1).

There is one current-value feature for every analogue and digital sensor. These
features preserve the instantaneous operating state while the rolling groups
below provide history.

## 5. Sampling-quality features

### `sampling_interval_sec`

The elapsed time in seconds since the previous raw observation. The first row
has no previous observation and is recorded as `0`. This exposes irregular
sampling directly to the model instead of silently assuming a perfect cadence.

### `gap_over_60sec`

A binary indicator equal to `1` when the previous observation is more than 60
seconds away. It is useful because a rolling window with fewer samples may mean
“the machine changed” or simply “the logger was quiet”.

## 6. Analogue rolling features

The pattern is:

```text
<raw_analogue_name>_<statistic>_<window>min
```

The configured time windows are 5, 20, and 60 minutes. They are time-based
windows, not “last N rows”; this matters because the source stream contains
gaps. The current row is included (`closed="both"`), and future rows are never
used.

For every analogue sensor and every window, the following statistics are
created:

| Statistic | Meaning | Example |
|---|---|---|
| `mean` | Average level during the window; describes operating point | `TP2_mean_20min` |
| `std` | Spread around the average; describes short-term instability | `Oil_temperature_std_60min` |
| `min` | Lowest observed value; captures a dip or low limit | `Reservoirs_min_5min` |
| `max` | Highest observed value; captures a peak or high limit | `Motor_current_max_20min` |
| `range` | `max - min`; captures total movement in the window | `TP3_range_5min` |

Interpretation example: a current TP2 value well above `TP2_mean_20min`, along
with a large `TP2_range_5min`, indicates a recent pressure change rather than
just a high but stable pressure level.

## 7. Digital rolling features

The pattern is:

```text
<digital_state>_mean_<window>min
<digital_state>_transitions_<window>min
```

The same 5-, 20-, and 60-minute time windows are used.

### State mean

`<state>_mean_<window>min` is the fraction of observations in the window for
which the state is on. A value near `1` means it was almost continuously active;
a value near `0` means it was mostly inactive. For example,
`COMP_mean_5min = 0.8` means 80% of the available observations in the last
five minutes had the compressor state on. Because this is an observation
fraction, it is not exactly the same as a percentage of wall-clock time when a
gap is present.

### State transitions

`<state>_transitions_<window>min` counts absolute changes between consecutive
observations in the window. For example, a value of `6` for
`Pressure_switch_transitions_20min` means the switch changed state six times in
the available 20-minute history. A high count can indicate cycling, chattering,
or unstable control behaviour.

## 8. History coverage features

The pattern is:

```text
history_observations_<window>min
```

This counts how many raw observations contributed to the corresponding causal
window. It is especially useful alongside the sampling-quality columns:

- a normal 5-minute period may contain roughly 30 observations at the nominal
  10-second cadence;
- a long communication gap can leave only a few observations in a nominal
  60-minute window;
- `history_ready_60min` applies the current minimum-count rule (30
  observations) to that 60-minute count.

These counts describe feature reliability. They can be used as predictors if a
modelling experiment explicitly wants the model to learn data-availability
effects, but the readiness flag itself is intended primarily as a pre-model
quality filter.

## 9. Selecting `X` and `y` for modelling

Choose exactly one horizon as the target for an experiment. For example, for a
20-minute warning model:

```python
target = "horizon_label_20min"
target_columns = [
    "horizon_label_5min",
    "horizon_label_10min",
    "horizon_label_20min",
    "horizon_label_30min",
]
non_feature_columns = {
    "timestamp",
    "in_failure_window",
    "failure_id",
    "history_ready_60min",
    *target_columns,
}

usable = df[df["history_ready_60min"] == 1].copy()
X = usable.drop(columns=non_feature_columns)
y = usable[target]
```

This leaves the 173 generated sensor/statistical features. The exact decision
to include or exclude `history_observations_*`, `sampling_interval_sec`, and
`gap_over_60sec` should be recorded as an experiment setting; they are valid
causal measurements, while the audit and target columns are not.

## 10. Leakage and evaluation rules

- Use only one horizon label as `y`; never include any horizon label as a
  predictor for another horizon experiment.
- Exclude `timestamp`, `failure_id`, and `in_failure_window` from `X`.
- Apply the `history_ready_60min` filter before fitting, and do not turn the
  flag into a hidden label.
- Fit scalers, feature selectors, resampling methods (for example SMOTE), and
  decision thresholds on training data/validation data only. Keep test rows
  untouched until the final evaluation.
- Report row-level precision/recall and event-level detection plus lead time;
  sparse warning labels make raw accuracy misleading.
