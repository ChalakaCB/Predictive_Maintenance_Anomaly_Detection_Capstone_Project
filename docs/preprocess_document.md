# MetroPT-3 preprocessing

## Purpose

`src/preprocess.py` is the single offline preprocessing entry point for the
MetroPT-3 air-compressor dataset. It combines data cleaning, failure-label
construction, chronological splitting, and causal sliding-window feature
engineering. The output is ready for the traditional machine-learning stage.

## Input and output

Each teammate keeps the raw download locally at:

```text
data/raw/MetroPT3(AirCompressor).csv
```

The default output is written directly to `data/processed/`:

```text
data/processed/
├── train.csv
├── val.csv
└── test.csv
```

The generated data is local-only and is excluded by `.gitignore`.

`docs/MetroPT3_ETL.ipynb` is retained as a reference notebook contributed by
the team. It documents the earlier exploratory ETL work that informed the
current script, but it is not executed by the production preprocessing
command and should not be treated as a second pipeline.

## What the pipeline does

1. Removes the leftover CSV index, validates the required sensor columns, parses
   timestamps, converts sensor values to numeric values, and sorts chronologically.
2. Adds the four documented failure intervals (F1–F4) and the
   `horizon_label_5min`, `horizon_label_10min`, `horizon_label_20min`, and
   `horizon_label_30min` targets.
3. Creates chronological train/validation/test partitions in memory. No random
   row shuffle is used.
4. Builds causal, time-based history features over 5-, 20-, and 60-minute
   windows. Only the current and past sensor readings are used.
5. Writes one feature CSV per split. The feature naming conventions and input
   selection rules are documented in `docs/metadata.md`.

## Default split

The default boundaries are:

| Split | Time range | Role |
|---|---|---|
| Train | Before `2020-06-05 09:30:00` | Fit model parameters; contains F1 and F2. |
| Validation | `2020-06-05 09:30:00` to before `2020-07-14 00:00:00` | Tune parameters and choose an operating threshold; contains F3. |
| Test | From `2020-07-14 00:00:00` onward | Final, untouched evaluation; contains F4. |

The 30-minute training buffer matches the longest configured warning horizon,
so the validation event's warning rows cannot leak into training.

## Running the pipeline

Run from the repository root:

```bash
python src/preprocess.py
```

The script intentionally has no command-line parameters. It always reads
`data/raw/MetroPT3(AirCompressor).csv` and writes to `data/processed/`, using
paths relative to the repository root so the same command works on every
teammate's machine.

The fixed warning horizons are 5, 10, 20, and 30 minutes. The fixed history
windows are 5, 20, and 60 minutes. The script writes only the final feature
files; any intermediate split is held in memory.

## Leakage and modelling rules

- Select model inputs using the rules in `docs/metadata.md`: keep the generated
  sensor/statistical feature groups and drop row metadata and target columns.
- Do not use timestamps, failure identifiers, countdowns, horizon labels, or
  failure-window flags as predictors.
- Fit scaling, feature selection, resampling, and thresholds using training and
  validation data only. Keep the test distribution untouched.
- Treat `history_ready_60min` as a data-quality gate, not a sensor feature.
- Evaluate both row-level metrics and event-level recall/lead time. Accuracy is
  not informative for these sparse warning labels.

## Known limitations

The dataset has four documented events and very few positive warning rows. F1 is
reported with a broad day-level interval, and all events represent the same
reported compressor failure type. Results are therefore a baseline for the
research workflow, not evidence of general performance across machines or
failure modes.
