# 05. Validation & Benchmarks

## Overview
*Testing strategy, statistical evaluation metrics, baseline models, and validation protocols.*

---

## Evaluation Metrics
1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
3. **Mean Absolute Percentage Error (MAPE)**:
   $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
4. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## Validation Protocols
- **Time-Series Split / Rolling Origin Validation**: Prevents data leakage.
- **Baseline Models**: Naive persistence ($y_{t+h} = y_t$), Seasonal persistence ($y_{t+h} = y_{t+h-24}$), Moving Average.
- **Automated Sanity & Smoke Tests**: Shape checking, numerical stability checks (NaN/Inf checks).
