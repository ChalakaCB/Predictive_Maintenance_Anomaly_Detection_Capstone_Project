# MetroPT-3 ETL documentation

This document describes the current, reproducible ETL for the MetroPT-3 air-compressor dataset. It is written for the next feature-engineering and modelling stages, so it records both what was generated and what must **not** be treated as ground truth.

## Current files

The folder contains three complementary artifacts:

| File | Purpose |
|---|---|
| MetroPT3(AirCompressor).csv | Original downloaded sensor data. |
| MetroPT3_ETL.ipynb | Explanatory notebook: each transformation and exploratory check is visible cell by cell. |
| MetroPT3_ETL.py | Repeatable local version of the ETL. It does not require Colab or Google Drive. |

The Python script is the reproducible source for regenerating the two files in ETL_OUTPUT Dataset. From this folder run:

~~~powershell
python MetroPT3_ETL.py
~~~

The default input and output paths are relative to the script. The existing CSVs in ETL_OUTPUT Dataset may have been generated before the 5/10/20-minute labels were added; rerunning the script refreshes them with all four horizons.

---

## 1. Raw data and quality checks

The raw file contains 1,516,948 rows and 17 columns, including one leftover row-index column. After removing that index there are 16 useful columns: a timestamp, seven analogue measurements, and eight digital state signals.

The nominal sampling interval is approximately 10 seconds, but it is not perfectly uniform. The ETL therefore sorts by timestamp and the modelling stage should inspect timestamp gaps before assuming that a row always equals exactly 10 seconds. The raw audit found no missing values and the digital columns contain binary 0/1 values. There are also long timestamp gaps, so rolling features should be interpreted as row-based windows unless a later stage explicitly resamples by time.

The useful raw columns are:

~~~
timestamp, TP2, TP3, H1, DV_pressure, Reservoirs,
Oil_temperature, Motor_current,
COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch,
Oil_level, Caudal_impulses
~~~

---

## 2. Loading, cleaning, and ordering

The notebook and script perform the following operations:

~~~python
df = pd.read_csv(input_path)
df.columns = [str(c).strip() for c in df.columns]

# The first column in the downloaded file is a leftover index.
if df.columns[0] != "timestamp":
    df = df.drop(columns=[df.columns[0]])

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp").reset_index(drop=True)
~~~

Sorting is essential: all failure windows, countdowns, rolling statistics, and the chronological train/test split depend on true time order.

---

## 3. Documented failure windows

The dataset does not provide a row-level failure target. The four documented events from the accompanying report are encoded as intervals:

| ID | Start | End | Role in this ETL |
|---|---|---|---|
| F1 | 2020-04-18 00:00 | 2020-04-18 23:59 | Broad day-level interval reported for the first event. |
| F2 | 2020-05-29 23:30 | 2020-05-30 06:00 | Failure interval. |
| F3 | 2020-06-05 10:00 | 2020-06-07 14:30 | Failure interval. |
| F4 | 2020-07-15 14:30 | 2020-07-15 19:00 | Most recent event, held out for testing. |

For every row the ETL creates:

- in_failure_window: 1/True when the timestamp is inside one of the intervals;
- failure_id: F1–F4 inside an interval and blank elsewhere.

These are event annotations for analysis. They are not valid model inputs: a live system would not know that a failure is currently happening when it is making a pre-failure warning.

The next-start countdown is computed as:

~~~python
failure_starts = [start_time for _, start_time, _ in FAILURES]
df["time_to_next_failure_min"] = minutes_until_the_next_strictly_future_start
~~~

The script uses a vectorised equivalent of the notebook's row-wise calculation. It is a label-construction helper and must be excluded from model features because it directly contains future information.

---

## 4. Four fixed warning horizons

The main research question is: **given the past sensor history, will a documented failure start within a selected future warning period?** The ETL now builds four separate binary targets:

~~~python
HORIZONS_MIN = [5, 10, 20, 30]

valid_pre_failure = (
    df["time_to_next_failure_min"].notna()
    & (df["time_to_next_failure_min"] > 0)
    & (~df["in_failure_window"])
)

for horizon in HORIZONS_MIN:
    df[f"horizon_label_{horizon}min"] = (
        valid_pre_failure
        & (df["time_to_next_failure_min"] <= horizon)
    ).astype("int8")
~~~

The strict > 0 condition means that a row at or after the failure start is not counted as a warning. Rows already inside a failure window are also excluded. Each horizon is a separate target; do not combine them into one label before evaluation.

With the current four intervals and split, the expected counts are:

| Target | All rows | Train (F1–F3) | Test (F4) |
|---|---:|---:|---:|
| horizon_label_5min | 115 | 85 | 30 |
| horizon_label_10min | 230 | 170 | 60 |
| horizon_label_20min | 462 | 341 | 121 |
| horizon_label_30min | 694 | 513 | 181 |

These positives are extremely sparse. A longer horizon gives more positive rows, but it is still a label derived from only four reported events, not a new set of independent failures.

---

## 5. Adaptive statistical diagnostics (weak labels)

The notebook also creates a second family of exploratory signals. They are useful for studying whether the compressor changes before a failure, but they are **not externally supplied ground-truth labels**.

### 5.1 Oil-temperature z-score

Oil_temperature is compared with its previous 720 rows (roughly two hours at the nominal 10-second cadence):

~~~python
roll_mean = df["Oil_temperature"].rolling(720, min_periods=100).mean().shift(1)
roll_std = df["Oil_temperature"].rolling(720, min_periods=100).std().shift(1)
df["oil_temp_zscore"] = (df["Oil_temperature"] - roll_mean) / roll_std
df["oil_temp_anomaly"] = (df["oil_temp_zscore"].abs() > 3).astype(int)
~~~

The shift is important: the baseline uses data before the current row rather than future observations.

### 5.2 Motor-off duration

The ETL groups consecutive rows with Motor_current < 0.1 into stretches, measures each stretch, and compares it with the previous 20 off-stretches. The result is off_duration_anomaly.

The current implementation mirrors the notebook by calculating a completed stretch's start and end before assigning the anomaly to all rows in that stretch. That is acceptable for exploratory retrospective analysis, but it is **not causal for an online streaming product**: a live system does not know the end of an ongoing stretch. This signal must be redesigned (for example, using elapsed off-time available at the current row) before it is used as a production feature.

### 5.3 Combined signal

~~~python
df["adaptive_statistical_label"] = (
    (df["oil_temp_anomaly"] == 1)
    | (df["off_duration_anomaly"] == 1)
).astype(int)
~~~

This is best described as a **weak, rule-based anomaly signal**. It can be used for exploratory plots, an ablation study, or a candidate early-warning feature after a causal rewrite. It should not be reported as a verified failure label, and the ETL does not claim that it provides a guaranteed number of hours of warning for every event.

---

## 6. Chronological train/test split

The split boundary is:

~~~python
split_boundary = pd.Timestamp("2020-07-14 00:00:00")
df["split"] = np.where(df["timestamp"] < split_boundary, "train", "test")
~~~

This gives:

- **Train:** 1,162,265 rows containing F1, F2, and F3;
- **Test:** 354,683 rows containing F4 only.

This is preferable to random row shuffling for a time-series warning task. Random shuffling would place nearly identical neighbouring readings on both sides of the split and would inflate scores. Holding out the latest failure tests whether a model can recognise a later event that was not used to fit the model.

The saved files do not keep the temporary split column; the file identity (metropt3_train.csv versus metropt3_test.csv) carries that information.

---

## 7. Output schema

Both output files contain the original sensor fields plus the following ETL fields.

### Original sensor fields

| Column group | Fields | Modelling role |
|---|---|---|
| Time | timestamp | Ordering, window construction, event-level evaluation; do not use the raw timestamp as an unrestricted numeric feature. |
| Analogue | TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current | Candidate sensor inputs. |
| Digital | COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses | Candidate state inputs. |

### ETL-created fields

| Field | Meaning | Default use |
|---|---|---|
| in_failure_window | Row is inside a documented failure interval. | Analysis only; exclude from features. |
| failure_id | F1–F4 inside a documented interval, blank elsewhere. | Event grouping and plots; exclude from features. |
| time_to_next_failure_min | Minutes to the next documented failure start. | Label helper/analysis only; direct future leakage. |
| horizon_label_5min | Failure starts within the next 5 minutes. | Short-warning target. |
| horizon_label_10min | Failure starts within the next 10 minutes. | Target. |
| horizon_label_20min | Failure starts within the next 20 minutes. | Recommended primary client-facing target. |
| horizon_label_30min | Failure starts within the next 30 minutes. | Longer comparison target. |
| oil_temp_zscore | Current oil temperature's look-back z-score. | Diagnostic or carefully validated causal feature. |
| oil_temp_anomaly | Absolute oil-temperature z-score exceeds 3. | Diagnostic/ablation feature. |
| off_duration_anomaly | Completed off-stretch is unusually long. | Exploratory only until causal redesign. |
| adaptive_statistical_label | Either statistical diagnostic fires. | Weak exploratory label, not ground truth. |

At present, warning rows have a blank failure_id because the event ID is only assigned inside the failure interval. The modelling/evaluation stage should create a separate target_failure_id from the next failure start if event-level recall and lead time are required; this should be kept out of the feature matrix.

---

## 8. Important limitations and leakage controls

1. **The four horizon targets are constructed labels.** The raw dataset has four documented event intervals, not a row-level failure_soon column. The 5/10/20/30-minute windows are explicit research definitions. F1 is reported at day-level resolution, so its exact onset is less precise than F2–F4.
2. **Only four events are available, and they represent the same reported air-leak failure type.** A strong score here cannot be presented as general performance for every compressor or pump fault.
3. **Do not use future-derived ETL fields as features.** Exclude time_to_next_failure_min, all horizon labels other than the selected target, in_failure_window, failure_id, and any post-event information.
4. **The adaptive off-duration signal is currently non-causal.** It uses the end of a completed stretch to label that stretch. Redesign it before claiming a real-time warning system.
5. **Build windows after the temporal split.** Compute rolling/window features independently inside train and test, or use only past context at each test timestamp. Do not calculate a window over the full table and then let it cross the split boundary.
6. **Fit preprocessing on train only.** Scaling, imputation choices, feature selection, and SMOTE must be learned/applied using training data only. Keep the test distribution at its natural severe imbalance.
7. **Use imbalance-aware and event-aware metrics.** Report precision, recall, F1, PR-AUC, false alarms per hour/day, event-level recall, and warning lead time. Plain accuracy is not informative for these sparse targets.

---

## 9. Handoff to feature engineering and modelling

The clean experimental design is to keep the same features and temporal split while changing only the target:

1. Build causal sliding-window features from the raw sensor columns (for example, 60-minute and 120-minute history, plus shorter windows if justified).
2. Train/evaluate one model per target: horizon_label_5min, horizon_label_10min, horizon_label_20min, and horizon_label_30min. A multi-output model can be an additional experiment, but the four results should remain separately interpretable.
3. Treat 20 minutes as the primary client-facing experiment, with 5 minutes as a short-horizon baseline and 10/30 minutes as sensitivity comparisons.
4. If SMOTE or another resampling method is tested, apply it to the training window-level samples only. Never resample the chronological test set.
5. Keep adaptive_statistical_label as a separate ablation/weak-supervision experiment. If it is used as an input, compare a model with and without it and document the causal rewrite first.
6. Evaluate by failure event as well as by row. With only F4 in the held-out period, report the uncertainty and avoid interpreting one event as broad generalisation evidence.

---

## 10. Re-running the ETL

Default run:

~~~powershell
cd D:\pump\metropt+3+dataset
python MetroPT3_ETL.py
~~~

Custom paths or horizons:

~~~powershell
python MetroPT3_ETL.py --input "D:\pump\metropt+3+dataset\MetroPT3(AirCompressor).csv" --output-dir "D:\pump\metropt+3+dataset\ETL_OUTPUT Dataset" --horizons 5 10 20 30
~~~

The command writes metropt3_train.csv and metropt3_test.csv to the selected output directory and prints the quality checks, label counts, failure allocation, and output shapes. The notebook remains useful for explanation and visual exploration; the script is the repeatable local build used to regenerate the handoff files.

