# 02. Dataset & Data Flow

## Overview
*Details about the dataset source, features, temporal resolution, preprocessing operations, and how data moves across the pipeline.*

---

## Dataset Profile
- **Dataset Source**: *(To be populated upon dataset selection/provision)*
- **Frequency / Granularity**: *(e.g., hourly, 15-minute, daily)*
- **Target Variable**: Energy Consumption (kWh / MWh)
- **Exogenous Variables**: Weather (Temperature, Humidity), Calendar Features (Day of week, Holiday, Season)

---

## Data Flow Pipeline
1. **Raw Ingestion**: Read raw time-series records.
2. **Data Cleaning**: Handling missing timestamps, outliers, interpolation.
3. **Feature Generation**: Lags, rolling statistics, Fourier cyclical time embeddings.
4. **Data Splitting**: Temporal train/validation/test split (no lookahead bias).
5. **Scaling & Normalization**: Fit on train set only, transform validation/test.
