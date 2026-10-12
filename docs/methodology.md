# Methodology & Modeling Strategy

This document details the planned methodology, modeling paradigms, validation scheme, and evaluation protocol for forecasting household electric power consumption.

---

## Forecasting Task Definition

- **Target Variable**: Hourly aggregated active energy consumption (`energy_kwh`), calculated from minute-level `Global_active_power` readings.
- **Forecast Horizon ($H$)**: Short-term horizon (e.g., 24 hours to 168 hours ahead).
- **Objective**: Accurately predict future consumption values based on historical patterns, seasonality, and autoregressive signals.

---

## ️ Feature Engineering Contract (Phase 3)

Feature generation is handled by `src/feature_engineering.py`. Generating features prepares input representations but **does not itself train or evaluate any model**.

### 1. Calendar & Temporal Features (`extract_temporal_features`)
Extracted directly from the verified hourly `timestamp`:
- `hour`: Hour of the day (0–23), capturing intraday diurnal load cycles.
- `day_of_week`: Day of the week (0=Monday, 6=Sunday), capturing weekly routine variations.
- `day_of_month`: Day of the month (1–31).
- `month`: Month of the year (1–12), capturing broad seasonal temperature/heating shifts.
- `is_weekend`: Binary flag (1 if Saturday or Sunday, else 0), separating residential occupancy regimes.

### 2. Autoregressive Lag Features (`create_lag_and_rolling_features`)
Captures serial dependency and load inertia:
- `lag_1`: Consumption at timestamp $T - 1\text{ hour}$.
- `lag_24`: Consumption at timestamp $T - 24\text{ hours}$ (daily seasonal lag).
- `lag_168`: Consumption at timestamp $T - 168\text{ hours}$ (weekly seasonal lag).

**Parameter Validation & Anti-Leakage Policy**:
- Lags must be a non-empty sequence of positive integers ($L \ge 1$, e.g. `(1, 24, 168)`).
- Zero, negative lags, booleans, and non-integers are strictly rejected with `ValueError`. Negative lags are prohibited because they introduce future target information (lookahead leakage).

**Missing Timestamp & Lag Policy**:
Because data cleaning may exclude low-coverage hours, observations in the processed series can be irregular. Lags are calculated using **exact timestamp-based lookups** ($T - L\text{ hours}$). If the exact historical timestamp is absent from the dataset, the lag is recorded as `NaN`. Naive positional shifting (`df.shift(1)`) is strictly prohibited to avoid associating observations from non-consecutive hours.

### 3. Rolling Window Aggregations
- `rolling_mean_24`: Mean consumption across the 24 hours immediately preceding timestamp $T$.

**Rolling Parameter Validation**:
- `rolling_window_hours` must be a positive integer ($W \ge 1$, default `24`). Non-integers, zero, negative numbers, and booleans are strictly rejected with `ValueError`.

**Target Leakage Prevention**:
The observation at timestamp $T$ is strictly excluded from its own rolling window calculation; the window covers only historical timestamps $[T - 24\text{h}, T - 1\text{h}]$.

**Rolling Completeness Policy**:
To prevent statistical distortion during periods with missing records, `rolling_mean_24` requires a complete set of 24 contiguous hourly observations. If any hour in $[T - 24\text{h}, T - 1\text{h}]$ is missing, `rolling_mean_24` evaluates to `NaN`. Incomplete windows are never filled with arbitrary constants or zero.

### 4. Input Timestamp Integrity & Duplicate Policy
- **Duplicate Timestamps Strictly Rejected**: Feature engineering functions reject inputs containing duplicate timestamps with a descriptive `ValueError`.
- **Deduplication Responsibility**: Duplicate detection and aggregation belong strictly to the preprocessing stage (`src/preprocessing.py`). Feature engineering will not silently take the first or average conflicting values, ensuring no ambiguity enters the feature generation pipeline.
- **Non-Null Timestamps**: Timestamps must be valid and non-null; missing or unparseable timestamps raise `ValueError`.

### 5. Row Preservation Policy
Rows with missing lag or rolling values are preserved in the engineered dataset. Row filtering and imputation are deferred to Phase 4 (model training), allowing different estimators (e.g., tree models vs. linear models) to apply their own documented row-selection policies.

---

---

## Modeling Candidates & Baselines (Phase 4A)

To ensure scientific rigor, all prospective machine learning models are benchmarked against standard time-series baselines:

### 1. Baseline A: Persistence (`predict_persistence`)
- **Policy**: For prediction timestamp $T$, predict using the observed target value at exactly $(T - 1\text{ hour})$.
- **Historical Lookup**: Lookups are performed against historical observations strictly prior to $T$.
- **Missing Timestamp Policy**: If the observation at $(T - 1\text{ hour})$ is missing or non-finite, returns `NaN`. Naive positional shifting is prohibited.

### 2. Baseline B: Daily Seasonal Naive (`predict_daily_seasonal_naive`)
- **Policy**: For prediction timestamp $T$, predict using the target observed exactly 24 hours prior: $(T - 24\text{ hours})$.
- **Missing Timestamp Policy**: If the exact observation at $(T - 24\text{ hours})$ is absent or non-finite, returns `NaN`. No substitute timestamp is used.

### 3. Machine Learning Model: Random Forest Regressor (`train_random_forest`)
- **Algorithm**: `RandomForestRegressor` from `scikit-learn`.
- **Reproducibility**: Deterministic training via fixed random seed (`random_state=42`).
- **Configuration**: `n_estimators=100`, `n_jobs=-1`.
- **Scaling**: No feature normalization or scaling applied (decision trees are invariant to monotonic scaling).
- **Fitting**: Fitted strictly on the training partition; validation and test partitions are never used during fitting.

---

## ⏱️ Chronological Validation Protocol (Phase 4A)

Time-series observations exhibit strong temporal dependencies and non-stationarity. Random cross-validation (K-Fold with shuffling) is strictly prohibited as it causes temporal leakage.

The project implements a strict chronological partitioning function (`chronological_split`):
1. **Partitioning Scheme**:
   - **Training Set**: Earliest ~70% of chronological records (`train_ratio=0.70`).
   - **Validation Set**: Subsequent ~15% (`val_ratio=0.15`) for model tuning and selection.
   - **Test Set**: Final ~15% held out strictly for final generalization evaluation.
2. **Temporal Integrity**:
   - Observations are sorted chronologically by timestamp before splitting.
   - Duplicate timestamps and null timestamps are strictly rejected with descriptive `ValueError`.
   - Partitions are mutually exclusive: $T_{\text{train, max}} < T_{\text{val, min}} \le T_{\text{val, max}} < T_{\text{test, min}}$.
   - Small datasets that cannot form non-empty partitions raise a clear `ValueError`.

---

## ️ Leakage Prevention & Feature Eligibility (Phase 4A)

### 1. Predictor Feature Selection (`select_predictor_features`)
Model predictors are explicitly selected to prevent target and temporal lookahead leakage:
- **Prohibited**: Raw `timestamp`, target `energy_kwh`, raw `Global_active_power`, and preprocessing metadata (`valid_minutes`, `coverage_ratio`).
- **Approved Predictors**:
  - Calendar features: `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`.
  - Autoregressive lag features: `lag_1`, `lag_24`, `lag_168`.
  - Rolling window features: `rolling_mean_24` (covering strictly $[T - 24\text{h}, T - 1\text{h}]$).

### 2. Partition-Level Row Eligibility (`filter_eligible_rows`)
- **Policy**: Supervised model fitting requires non-null target and predictor values.
- **Independence**: Eligibility filtering is performed independently within each chronological partition.
- **Index Preservation**: Eligible partitions strictly retain their original DataFrame indices (no index resetting), ensuring predictions map back to the exact physical timestamps and ground-truth rows without positional drift.
- **Zero-Imputation Prohibition**: Missing lag and rolling values are **never filled with zero or arbitrary constants**.
- **Accounting**: Every partition records total rows, eligible rows, excluded rows, and per-feature missing counts.

---

## Evaluation Metrics & Benchmarking Protocol (Phase 4A)

Performance is measured using standardized metrics computed via `compute_metrics` and `evaluate_and_compare_models`:

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
   Reported in physical units (kWh).

2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
   Penalizes large outlier forecast errors in kWh.

3. **Coefficient of Determination ($R^2$)**:
   Quantifies the proportion of variance explained by the model relative to a mean baseline. Handled safely when sample count $< 2$ or target variance is zero.

### Index-Aware Prediction Alignment (`align_predictions_to_dataframe`)
- Model prediction Series are explicitly validated against the evaluation DataFrame's index.
- Differently indexed or permuted Series are reindexed by key, not position.
- Disjoint indices or foreign keys trigger immediate `ValueError` to prevent scoring the wrong observation timestamps.

### Fair Comparison on Common Timestamps
Because baseline models require specific historical lookbacks ($T - 1\text{h}$, $T - 24\text{h}$) and feature engineering requires lookbacks up to $T - 168\text{h}$, models can have differing coverage.
The evaluation engine reports:
1. **Individual Coverage**: Metrics scored on all valid predictions for each model.
2. **Common Timestamps**: Metrics scored on the exact subset of timestamps where **all** evaluated models generated valid predictions, guaranteeing fair head-to-head comparison.

> **Data Availability Status**: Real-world model training and evaluation remain blocked because the raw UCI dataset has not been placed in `data/raw/household_power_consumption.txt`. The entire pipeline and modeling logic have been deterministically verified using synthetic test suites. No synthetic metrics are claimed as real-world results.

