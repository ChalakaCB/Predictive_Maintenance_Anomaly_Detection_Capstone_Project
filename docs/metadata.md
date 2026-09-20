# MetroPT-3 feature and column guide

This document describes the schema produced by `src/preprocess.py`. It is the
human-readable reference for the final CSV files; no separate machine-readable
metadata file is required. The same constants are defined in
`src/preprocess.py` so that the feature names and target definitions stay
reproducible.

## 1. What one output row represents

The initial feature pool is calculated in timestamp order. Each original
feature row uses the current observation and earlier observations only. The
final CSVs contain a mixture of original feature rows and SMOTE-generated
positive feature rows, after negative undersampling and random splitting.
They are not a chronological sensor stream.

There is one folder for each prediction horizon, with three files per folder:

```text
data/processed/
├── horizon_label_20min/
├── horizon_label_30min/
├── horizon_label_40min/
└── horizon_label_50min/
    └── train.csv, val.csv, test.csv    # These three files exist in every folder
```

These are the only preprocessing outputs: 12 CSVs in total. Each file has
174 columns: 173 model features and the selected horizon's one binary target.
There are no exported timestamps, event identifiers, other horizon labels,
readiness flags, or `split` columns.

| Group | Count | Purpose |
|---|---:|---|
| Current readings | 15 | Latest values from seven analogue and eight digital sensors |
| Sampling quality | 2 | Observation interval and gap indicator |
| Analogue rolling features | 105 | Seven sensors × five statistics × three windows |
| Digital rolling features | 48 | Eight sensors × two statistics × three windows |
| History observation counts | 3 | Available observations in each history window |
| Selected warning target | 1 | One of the 20-, 30-, 40-, or 50-minute labels |

For each target, rows without enough history are removed first. SMOTE brings
the positive count to twice its original eligible count; random undersampling
keeps up to 600 negatives per resulting positive, capped at the available
negative count. The resulting pool is shuffled and stratified into approximately
70% train, 15% validation, and 15% test, using random seed 42. The current data
support the requested positive:negative ratio of 1:600 for all four targets;
individual split counts can differ slightly because of integer rounding.

The feature interpretations below refer to the original, timestamp-based
features. SMOTE interpolates these numerical columns, so synthetic rows can
contain fractional digital states, gap indicators, transition counts, and
history counts. Such values are interpolated feature values, not new physical
observations. A synthetic row has no genuine observation timestamp or canonical
position in the original sequence.

## 2. Original sensor columns (grouped overview)

The raw dataset contains two practical sensor families. The raw names are kept
inside the generated feature names, but are grouped here rather than repeated
one by one.

### Analogue process measurements

These are continuous physical measurements:

- pressure-related readings (`TP2`, `TP3`, `H1`, `DV_pressure`);
- reservoir-related measurement (`Reservoirs`);
- temperature (`Oil_temperature`);
- electrical load (`Motor_current`).

They are suitable for level, variability, range, and trend-like context. For
example, a rising `TP2` rolling mean or a large `Oil_temperature` rolling
standard deviation can describe a change in operating behaviour.

### Digital operating states

These are discrete operating or switch signals, normally represented as 0/1:
compressor state, electric valve, cooling towers, motor protection/impulse
signals, pressure switch, oil-level switch, and flow impulses. Their rolling
features describe the fraction of observations where a state is active and how often it
changes, rather than treating the state as a continuous physical quantity.

## 3. Internal audit fields and the exported target

The fields below are used while constructing labels and causal features.
Only the selected `horizon_label_<N>min` is exported as `y`; the audit fields,
readiness flag, and all target columns are excluded from the feature matrix
before SMOTE. None of these fields belongs in the model's 173 inputs.

### `timestamp`

The original observation time, used for ordering, time-based windows, and
matching documented failure intervals. It is not exported in the final CSVs
and is not passed to the classifier.

### `in_failure_window`

`True` when the timestamp lies inside one of the four documented failure
intervals (F1-F4). Rows in these intervals are not warning examples. The
intervals represent the reported maintenance/failure periods, not a sensor
prediction made by the pipeline.

### `failure_id`

The event identifier (`F1`, `F2`, `F3`, or `F4`) for rows inside a documented
failure interval; otherwise it is empty. It is internal context and is not
exported. It cannot be used to reconstruct event-level results from the final
randomly sampled CSVs alone.

### `horizon_label_<N>min`

There is one binary target for each `N` in `{20, 30, 40, 50}`. A value is `1`
when the next documented failure starts more than 0 and no more than `N`
minutes after the current timestamp, and the current row is outside a failure
window. Otherwise it is `0`.

For example, if F4 starts at 14:30, a row at 13:55 has only the 40- and
50-minute labels equal to `1` (it is 35 minutes before the event), while a row
at 14:12 has all four labels equal to `1` (it is 18 minutes before the event).
A row inside F4 has all warning labels equal to `0`.

Each final CSV contains only its own folder's target, not all four labels.
The internal countdown used to construct the labels is not exported. Rows
already inside a failure interval remain negative: this target describes an
upcoming failure onset, not the current faulty state. SMOTE adds positive
feature rows with label `1`; labels remain binary.

### `history_ready_60min`

This is a quality-control flag, not a sensor feature. It is `1` when at least
30 observations are available in the causal 60-minute context and `0` when
fewer are available, for example during initial warm-up or after a long gap.
This is a minimum observation-count rule, not a requirement to have a full
60-minute elapsed history or an uninterrupted hour of readings. Rows with `0`
are removed before resampling, and the flag is then dropped. No additional
readiness filtering is required when loading a final modelling CSV.

## 4. Current-value features

The pattern is:

```text
current_<raw_sensor_name>
```

For an original feature row, it is the latest reading available at that
timestamp, with no look-ahead. Examples:

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

`<state>_transitions_<window>min` sums absolute changes between consecutive
observations, assigning each change to the later observation's timestamp.
The first observation inside a window can therefore include a change from
its predecessor just outside the window. For example, a value of `6` for
`Pressure_switch_transitions_20min` means the switch changed state six times in
the available 20-minute history. A high count can indicate cycling, chattering,
or unstable control behaviour.

## 8. History coverage features

The pattern is:

```text
history_observations_<window>min
```

For an original feature row, this counts how many raw observations contributed
to the corresponding causal window. It is especially useful alongside the
sampling-quality columns:

- a normal 5-minute period may contain roughly 30 observations at the nominal
  10-second cadence;
- a long communication gap can leave only a few observations in a nominal
  60-minute window;
- `history_ready_60min` applies the current minimum-count rule (30
  observations) to that 60-minute count.

These three count columns are included in the 173 model inputs and describe
data availability. SMOTE can make them fractional in synthetic rows. The
readiness flag is different: it is used only to filter the original feature
pool and is not exported.

## 9. Selecting `X` and `y` for modelling

Choose exactly one horizon as the target for an experiment. For example, for a
20-minute warning model:

```python
import pandas as pd

target = "horizon_label_20min"
df = pd.read_csv(f"data/processed/{target}/train.csv")
X = df.drop(columns=[target])
y = df[target]
```

This leaves exactly 173 features, including `history_observations_*`,
`sampling_interval_sec`, and `gap_over_60sec`. Load `val.csv` and `test.csv` from
the same target folder and preserve the training feature names and order.
Fit any model-specific standardization on `X_train` only; apply the fitted
transform to validation, test, and deployment inputs. Do not resample these
already prepared CSVs again as part of the default modelling workflow.

## 10. Evaluation scope and chronological replay

The window calculations are causal, but that does not make the final splits
independent. The current preprocessing applies SMOTE to the full eligible pool
before splitting, so synthetic samples and their source neighbours can span
train, validation, and test. Adjacent, overlapping original windows can also
fall into different splits. These effects can inflate offline classification
metrics; the current random-split results do not establish performance on
unseen failure events.

During modelling, use the selected label only as `y`, select hyperparameters
and thresholds on validation data, and evaluate the chosen model on test data.
These choices do not remove the earlier full-pool resampling limitation.

For the display, replay the raw observations in chronological order and compute
the same features using current and past data, with the same history-count
gate. Then apply the saved model's feature order, standardization, and threshold.
Do not replay the shuffled modelling CSVs as if they were a real time series,
or run SMOTE during streaming inference. Replaying records used in training
demonstrates the product flow, not independent predictive performance.
