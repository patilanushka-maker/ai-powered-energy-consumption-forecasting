# Project Limitations & Scope

This document explicitly defines the boundaries, constraints, and operational assumptions of this energy forecasting project.

---

## ️ Core Limitations

### 1. Prototype Scope vs. Industrial Systems
- The dataset used is the public **UCI Individual Household Electric Power Consumption** benchmark, representing a single French residence (2006–2010).
- **Non-Generalizability**: Patterns observed in a single residential dwelling (domestic appliance schedules, residential heating/cooling cycles) **do not generalize** to commercial facilities, multi-tenant buildings, or industrial manufacturing energy profiles.
- This repository is designed as an educational and proof-of-concept forecasting prototype, not a certified industrial energy management system (EMS).

### 2. Inherent Forecast Uncertainty
- Energy demand is influenced by stochastic real-world factors (unpredicted human occupancy, appliance usage spikes, atypical weather extremes).
- All point forecasts carry epistemic and aleatoric uncertainty; predictions must be interpreted as probabilistic estimates rather than exact guarantees.

### 3. Multi-Step Error Accumulation
- In autoregressive recursive forecasting, errors from predicting step $t+1$ feed back as inputs for step $t+2$ through $t+H$.
- Over longer forecast horizons (e.g., beyond 48 hours), recursive forecast drift and variance compounding can degrade accuracy.

### 4. Tariff and Cost Estimation Approximations
- Any simulated financial or tariff calculations (such as time-of-use pricing or peak demand charges) rely on simplified proxy assumptions.
- Actual utility billing structures entail complex demand ratchets, network charges, taxes, and fluctuating dynamic spot market prices that are not fully captured by static tariff approximations.

### 5. Forecasting Does Not Equal Energy Savings
- Generating a forecast prediction does not, by itself, reduce kilowatt-hour consumption or carbon footprint.
- Realized energy savings require downstream operational actions - such as load shifting, peak shaving, automated demand response, or physical equipment efficiency interventions.

### 6. Baseline and Historical Lookup Constraints (Phase 4A)
- Persistence and seasonal naive baselines require exact historical observations ($T - 1\text{h}$ and $T - 24\text{h}$). During periods following missing or excluded hours, these baselines yield missing predictions (`NaN`).
- Model evaluation must explicitly account for varying observation counts across models rather than comparing on disparate subsets.

### 7. Tree Model Extrapolation Limitations (Phase 4A)
- Random Forest Regressors predict target values as averages of terminal leaf values from training data.
- Consequently, tree ensembles **cannot extrapolate** to energy consumption values higher or lower than the extremes observed in the training partition (e.g., during unprecedented heatwaves, extreme cold snaps, or sudden household expansions).

### 8. Data Ingestion Prerequisite
- Real-world model training and empirical evaluation remain blocked until the raw UCI dataset is downloaded and extracted into `data/raw/household_power_consumption.txt`.
- No claims of real-world forecast accuracy are made prior to running the training pipeline on real historical records.

### 9. CLI Evaluation Stage Retraining
- In Phase 4A, invoking `python main.py --stage evaluate` executes the integrated training and evaluation pipeline (`run_train_stage`) rather than loading a pre-trained model artifact from disk in isolation.
- Decoupling standalone model artifact evaluation without re-fitting is scheduled for pipeline operationalization in subsequent phases.


