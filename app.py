"""AI-Powered Energy Consumption Forecasting Dashboard (Phase 5).

Streamlit Web Interface providing:
1. Overview: System architecture, dataset provenance, and headline performance.
2. Historical Consumption: Interactive time-series explorer with date filtering.
3. Forecasting: Live multi-step recursive forecasting rollout (1–168h) with CSV export.
4. Model Evaluation: Benchmarks comparing Random Forest against Persistence & Seasonal Naive.
5. Recursive Backtesting: Multi-horizon rolling origin evaluation (1h, 6h, 24h, 48h).
6. Data Quality & Limitations: Single-household caveats, coverage gating, and error accumulation.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard_utils import (
    compute_summary_statistics,
    filter_data_by_date_range,
    load_metrics_report,
    load_processed_data,
    prepare_forecast_plot_data,
)
from src.forecast import generate_forecast, load_forecasting_model

# Constants & Paths
DATA_PATH = Path("data/processed/hourly_energy.csv")
MODEL_PATH = Path("models/random_forest_model.joblib")
PHASE4A_METRICS_PATH = Path("outputs/metrics/phase4a_metrics.json")
BACKTEST_METRICS_PATH = Path("outputs/metrics/recursive_backtest_metrics.json")


# =========================================================================
# Cached Resource and Data Loaders
# =========================================================================

@st.cache_data(show_spinner="Loading processed energy dataset...")
def get_historical_data() -> pd.DataFrame:
    """Load and cache processed hourly energy dataset."""
    return load_processed_data(DATA_PATH)


@st.cache_resource(show_spinner="Loading serialized Random Forest model...")
def get_model_artifact() -> Dict[str, Any]:
    """Load and cache serialized model artifact."""
    return load_forecasting_model(MODEL_PATH)


@st.cache_data(show_spinner="Loading Phase 4A benchmark metrics...")
def get_phase4a_metrics() -> Dict[str, Any]:
    """Load and cache Phase 4A evaluation report."""
    return load_metrics_report(PHASE4A_METRICS_PATH)


@st.cache_data(show_spinner="Loading Phase 4B backtest metrics...")
def get_backtest_metrics() -> Dict[str, Any]:
    """Load and cache Phase 4B recursive backtest report."""
    return load_metrics_report(BACKTEST_METRICS_PATH)


# =========================================================================
# Page Setup
# =========================================================================

def setup_page_configuration() -> None:
    """Configure Streamlit page layout and title."""
    st.set_page_config(
        page_title="AI Energy Forecasting",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# =========================================================================
# Section 1: Overview
# =========================================================================

def render_overview_section() -> None:
    """Render Project Overview and Executive Summary."""
    st.title("⚡ AI-Powered Energy Consumption Forecasting")
    st.markdown(
        "**Production-grade machine learning forecasting system for residential electricity demand.**"
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Dataset Scope", value="34,143 Hours", help="Dec 2006 to Nov 2010 (~4 years)")
    with col2:
        st.metric(label="Test MAE (RF)", value="0.3344 kWh", delta="-13.0% vs Persistence", delta_color="inverse")
    with col3:
        st.metric(label="Test RMSE (RF)", value="0.4791 kWh", delta="-17.7% vs Persistence", delta_color="inverse")
    with col4:
        st.metric(label="Test Variance Explained", value="53.9% (R²=0.539)", help="On 4,865 common held-out timestamps")

    st.markdown("### System Context & Provenance")
    st.write(
        """
        - **Dataset Provenance**: Real minute-level active power readings from the official **UCI Machine Learning Repository**
          (*Individual Household Electric Power Consumption*), recorded at a single private home in Sceaux, France between December 2006 and November 2010.
        - **Aggregation Policy**: Raw minute observations are resampled to top-of-the-hour **energy consumption in kilowatt-hours (kWh)**:
          $$E_{\\text{hourly}} = \\bar{P}_{\\text{active}} \\times 1.0\\text{ hour}$$
          Hours with $< 90\\%$ valid minutes (54 minutes) are strictly excluded (446 intervals excluded across 4 years).
        - **Machine Learning Architecture**: Random Forest Regressor (100 trees, `random_state=42`) fitted strictly on temporal features,
          autoregressive lags ($t-1\\text{h}$, $t-24\\text{h}$, $t-168\\text{h}$), and a trailing 24-hour rolling mean.
        """
    )

    st.info(
        "ℹ️ **Historical Context**: All data and forecasts originate from 2006–2010. "
        "This system demonstrates historical time-series forecasting integrity and is **not** generating live predictions for the present year (2026)."
    )


# =========================================================================
# Section 2: Historical Consumption
# =========================================================================

def render_historical_section(df: pd.DataFrame) -> None:
    """Render interactive historical energy consumption explorer."""
    st.header("📈 Historical Energy Consumption Explorer")
    st.caption("Inspect 34,143 verified hourly energy records (December 2006 – November 2010).")

    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()

    # Sidebar / in-page date controls
    st.markdown("#### Select Date Range")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        start_date = st.date_input("Start Date", value=max_date - pd.Timedelta(days=14), min_value=min_date, max_value=max_date)
    with c2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    with c3:
        st.write("")
        st.write("")
        reset_recent = st.button("Reset to Last 14 Days")
        if reset_recent:
            start_date = max_date - pd.Timedelta(days=14)
            end_date = max_date

    try:
        filtered_df = filter_data_by_date_range(df, start_date, end_date)
    except Exception as exc:
        st.error(f"Error filtering dates: {exc}")
        return

    if len(filtered_df) == 0:
        st.warning("No observations found within the selected date window. Please adjust the dates.")
        return

    stats = compute_summary_statistics(filtered_df, "energy_kwh")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Selected Hours", f"{int(stats['count']):,}")
    m2.metric("Total Consumed", f"{stats['total_kwh']:,.1f} kWh")
    m3.metric("Average Load", f"{stats['mean_kwh']:.3f} kWh/h")
    m4.metric("Peak Demand", f"{stats['max_kwh']:.3f} kWh/h")
    m5.metric("Minimum Demand", f"{stats['min_kwh']:.3f} kWh/h")

    # Time-series Chart
    st.markdown("#### Hourly Load Curve")
    chart_data = filtered_df.set_index("timestamp")[["energy_kwh"]].rename(columns={"energy_kwh": "Energy Consumption (kWh)"})
    st.line_chart(chart_data, color="#3182CE")

    with st.expander("🔍 View Raw Observations Table"):
        st.dataframe(filtered_df.sort_values("timestamp", ascending=False), width="stretch")


# =========================================================================
# Section 3: Forecasting
# =========================================================================

def render_forecasting_section(df: pd.DataFrame, model_payload: Dict[str, Any]) -> None:
    """Render interactive multi-step recursive forecasting generator."""
    st.header("⚡ Multi-Step Recursive Forecasting")
    st.caption("Generate forward predictions using the trained Random Forest model and verified historical data.")

    # Controls
    if "forecast_horizon" not in st.session_state:
        st.session_state["forecast_horizon"] = 24

    def set_horizon(hours: int) -> None:
        st.session_state["forecast_horizon"] = hours

    col_ctrl, col_info = st.columns([3, 4])
    with col_ctrl:
        st.markdown("#### Select Forecast Horizon")

        preset_col1, preset_col2, preset_col3 = st.columns(3)
        preset_col1.button("24 Hours (1 Day)", on_click=set_horizon, args=(24,))
        preset_col2.button("48 Hours (2 Days)", on_click=set_horizon, args=(48,))
        preset_col3.button("168 Hours (7 Days)", on_click=set_horizon, args=(168,))

        horizon = st.slider(
            "Horizon in Hours",
            min_value=1,
            max_value=168,
            key="forecast_horizon",
            step=1,
            help="Number of future consecutive hours to predict recursively (1 hour to 7 days)."
        )

    with col_info:
        latest_ts = df["timestamp"].iloc[-1]
        model_name = model_payload.get('model_type', 'RandomForestRegressor')
        lookback_start = latest_ts - pd.Timedelta(hours=167)
        pred_start = latest_ts + pd.Timedelta(hours=1)
        pred_end = latest_ts + pd.Timedelta(hours=horizon)

        st.markdown("#### Forecast Origin Metadata")
        st.markdown(
            f"- **Forecast Origin (T_origin)**: `{latest_ts}`\n"
            f"- **Model Architecture**: `{model_name}` (100 trees)\n"
            f"- **Historical Lookback**: 168 contiguous hours (`{lookback_start}` -> `{latest_ts}`)\n"
            f"- **Prediction Range**: `{pred_start}` to `{pred_end}`"
        )

    # Execution Button
    if st.button("🚀 Generate Recursive Forecast", type="primary"):
        with st.spinner(f"Computing {horizon}-hour recursive autoregressive rollout..."):
            try:
                forecast_df = generate_forecast(
                    model_or_path=model_payload,
                    history=df,
                    horizon_hours=horizon,
                )
            except Exception as exc:
                st.error(f"❌ Forecasting failed: {exc}")
                return

        st.success(f"Successfully generated {len(forecast_df)} hourly predictions starting at `{forecast_df['timestamp'].iloc[0]}`!")

        # Chart: Recent History + Forecast Connected
        st.markdown("#### Trajectory Visualization (Preceding 48h Actuals + Future Forecast)")
        try:
            plot_df = prepare_forecast_plot_data(df, forecast_df, lookback_hours=48)
            plot_chart_data = plot_df.set_index("timestamp")[["actual_energy_kwh", "forecast_energy_kwh"]].rename(
                columns={
                    "actual_energy_kwh": "Historical Actuals (kWh)",
                    "forecast_energy_kwh": "Model Forecast (kWh)",
                }
            )
            st.line_chart(plot_chart_data, color=["#718096", "#E53E3E"])
        except Exception as exc:
            st.warning(f"Could not build combined trajectory: {exc}")
            st.line_chart(forecast_df.set_index("timestamp")[["forecast_energy_kwh"]])

        # Summary KPIs of generated forecast
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted Total Energy", f"{forecast_df['forecast_energy_kwh'].sum():.2f} kWh")
        k2.metric("Mean Hourly Demand", f"{forecast_df['forecast_energy_kwh'].mean():.3f} kWh")
        k3.metric("Peak Forecasted Hour", f"{forecast_df['forecast_energy_kwh'].max():.3f} kWh")
        k4.metric("Base Forecasted Load", f"{forecast_df['forecast_energy_kwh'].min():.3f} kWh")

        # Table & CSV Download
        c_table, c_csv = st.columns([3, 1])
        with c_table:
            st.markdown("#### Forecast Predictions Table")
            st.dataframe(forecast_df, width="stretch", height=250)
        with c_csv:
            st.markdown("#### Export Results")
            csv_data = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_data,
                file_name=f"energy_forecast_{horizon}h.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.warning(
        "⚠️ **Recursive Rollout Mechanism**: Each step $s$ feeds its own predicted value $\\hat{y}_s$ back into the history buffer "
        "to serve as $\\text{lag}_1$, $\\text{lag}_{24}$, and rolling average inputs for step $s+1$. "
        "Consequently, prediction errors naturally accumulate over longer horizons ($H > 24\\text{h}$)."
    )


# =========================================================================
# Section 4: Model Evaluation
# =========================================================================

def render_evaluation_section(metrics: Dict[str, Any]) -> None:
    """Render Phase 4A Model Benchmark Scorecards."""
    st.header("📊 Model Evaluation & Benchmark Scorecard")
    st.caption("Chronologically partitioned benchmark comparisons from Phase 4A.")

    split_sum = metrics.get("split_summary", {})
    st.markdown("#### Chronological Partitioning (No Temporal Leakage)")
    p1, p2, p3 = st.columns(3)
    p1.info(f"**Training Set (70.0%)**\n\n`{split_sum.get('train_start')}` to `{split_sum.get('train_end')}`\n\n**23,900 Rows** (23,345 eligible)")
    p2.info(f"**Validation Set (15.0%)**\n\n`{split_sum.get('val_start')}` to `{split_sum.get('val_end')}`\n\n**5,121 Rows** (4,985 eligible)")
    p3.info(f"**Held-out Test Set (15.0%)**\n\n`{split_sum.get('test_start')}` to `{split_sum.get('test_end')}`\n\n**5,122 Rows** (4,865 common eligible)")

    st.markdown("#### Test Set Performance Benchmark (4,865 Identical Common Timestamps)")

    test_eval = metrics.get("test_evaluation", {})
    common_evals = test_eval.get("common_timestamp_evaluations", {})

    table_records = []
    for model_name, score_dict in common_evals.items():
        table_records.append(
            {
                "Model": model_name,
                "MAE (kWh)": f"{score_dict.get('mae'):.4f}",
                "RMSE (kWh)": f"{score_dict.get('rmse'):.4f}",
                "R² Score": f"{score_dict.get('r2'):.4f}",
                "Scored Common Samples": f"{score_dict.get('scored_common_samples'):,}",
            }
        )

    score_df = pd.DataFrame(table_records)
    st.table(score_df)

    st.markdown("#### Key Benchmark Takeaways")
    st.write(
        """
        1. **Random Forest vs. Persistence (1h Naive)**: Random Forest delivers a **13.0% reduction in MAE** (0.3344 vs. 0.3844 kWh)
           and increases $R^2$ from 0.3199 to **0.5390**.
        2. **Random Forest vs. Seasonal Naive (24h Naive)**: Seasonal Naive exhibits negative $R^2$ (-0.1443) due to day-to-day shifts
           in household occupancy and discretionary appliance cycles. Random Forest outperforms Seasonal Naive by **34.2% in MAE**.
        3. **Fair Evaluation Guarantee**: All models in this table were evaluated on the **exact same 4,865 timestamps** to prevent survivorship bias.
        """
    )


# =========================================================================
# Section 5: Recursive Backtesting
# =========================================================================

def render_backtest_section(backtest_metrics: Dict[str, Any]) -> None:
    """Render Phase 4B Recursive Backtest Evaluation."""
    st.header("🔄 Multi-Horizon Recursive Backtesting")
    st.caption("Rolling-origin backtest evaluating recursive multi-step forecasting across the held-out test split.")

    accounting = backtest_metrics.get("origin_accounting", {})
    origin_range = backtest_metrics.get("origin_range", {})

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Candidate Origins", f"{accounting.get('total_candidate_origins')}")
    b2.metric("Eligible Evaluated Origins", f"{accounting.get('eligible_origins')}")
    b3.metric("Skipped Origins", f"{accounting.get('skipped_origins_total')}", help="7 due to historical gaps, 2 due to forward target gaps")
    b4.metric("Origin Stride", f"{origin_range.get('origin_stride_hours')} Hours", help="72h stride exceeds 48h horizon -> Disjoint windows")

    st.markdown("#### Multi-Horizon Error Progression")

    horizon_metrics = backtest_metrics.get("horizon_metrics", {})
    horizon_rows = []
    for h_key, h_data in horizon_metrics.items():
        h_hours = h_data.get("horizon_hours")
        rf_mae = h_data.get("random_forest", {}).get("mae")
        rf_rmse = h_data.get("random_forest", {}).get("rmse")
        p_mae = h_data.get("persistence", {}).get("mae")
        p_rmse = h_data.get("persistence", {}).get("rmse")
        improvement = h_data.get("rf_mae_improvement_pct")

        horizon_rows.append(
            {
                "Horizon": f"{h_hours} Hours",
                "RF MAE (kWh)": f"{rf_mae:.4f}",
                "RF RMSE (kWh)": f"{rf_rmse:.4f}",
                "Persistence MAE (kWh)": f"{p_mae:.4f}",
                "Persistence RMSE (kWh)": f"{p_rmse:.4f}",
                "RF MAE Improvement": f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%",
            }
        )

    st.table(pd.DataFrame(horizon_rows))

    st.markdown("#### Horizon Interpretation")
    st.write(
        """
        - **Why Horizon 48h MAE (0.2783 kWh) is lower than Horizon 24h MAE (0.3650 kWh)**:
          This does **not** mean the model becomes more accurate further in the future. Rather, residential energy consumption exhibits
          pronounced diurnal variation. Hourly demand variance at 48 hours happens to land on hours with lower baseline volatility
          across the evaluated sampling grid.
        - **Disjoint Window Independence**: Because the origin stride ($72\\text{h}$) strictly exceeds the maximum evaluation horizon ($48\\text{h}$),
          consecutive forecast windows do not overlap. Every target timestamp in the backtest is evaluated exactly once without residual autocorrelation.
        """
    )


# =========================================================================
# Section 6: Data Quality and Limitations
# =========================================================================

def render_limitations_section() -> None:
    """Render explicit data quality policies, caveats, and operational limitations."""
    st.header("⚠️ Data Quality & Operational Limitations")

    st.markdown("### 1. Single Household Scope & Generalization")
    st.warning(
        "**Strict Single-Household Constraint**: The dataset captures electrical load from a single residence in Sceaux, France (2006–2010). "
        "These results demonstrate the algorithmic and temporal validity of the Random Forest model for this specific household. "
        "They **cannot** be generalized to other residences, commercial buildings, or different geographic regions without local retraining."
    )

    st.markdown("### 2. Historical Data vs. Real-Time Telemetry")
    st.write(
        """
        - **No Live Telemetry**: This project processes historical data terminating on **November 26, 2010**. Predictions generated
          represent historical out-of-sample experiments, not real-time 2026 grid forecasts.
        - **Contiguous History Requirement**: Generating lag and rolling features requires a strictly contiguous 168-hour historical window.
          In an operational production environment, telemetry dropouts must be handled via fallback models (e.g. calendar-only models).
        """
    )

    st.markdown("### 3. Missing Data & Hourly Completeness Policy")
    st.write(
        """
        - During preprocessing, hours meeting the $\\ge 90\\%$ valid-minute threshold (54+ minutes) are resampled using the observed mean active power.
        - Hours falling below 90% coverage are excluded entirely (446 intervals across the 4-year span).
        - Unobserved hours are **never** filled with zero or synthetically fabricated, ensuring the ML model is trained exclusively on genuine physical consumption.
        """
    )


# =========================================================================
# Main Application Entry Point
# =========================================================================

def main() -> None:
    """Orchestrate Streamlit dashboard layout and tab navigation."""
    setup_page_configuration()

    # Load resources safely with graceful fallback
    try:
        df = get_historical_data()
    except Exception as exc:
        st.error(f"❌ Critical Failure: Unable to load historical dataset: {exc}")
        return

    try:
        model_payload = get_model_artifact()
    except Exception as exc:
        st.error(f"❌ Critical Failure: Unable to load trained model artifact: {exc}")
        return

    try:
        phase4a_metrics = get_phase4a_metrics()
    except Exception as exc:
        st.error(f"❌ Critical Failure: Unable to load Phase 4A metrics: {exc}")
        return

    try:
        backtest_metrics = get_backtest_metrics()
    except Exception as exc:
        st.error(f"❌ Critical Failure: Unable to load backtest metrics: {exc}")
        return

    # Navigation Tabs
    tab_overview, tab_history, tab_forecast, tab_eval, tab_backtest, tab_limits = st.tabs(
        [
            "🏠 Overview",
            "📈 Historical Consumption",
            "⚡ Forecasting",
            "📊 Model Evaluation",
            "🔄 Recursive Backtest",
            "⚠️ Quality & Limitations",
        ]
    )

    with tab_overview:
        render_overview_section()

    with tab_history:
        render_historical_section(df)

    with tab_forecast:
        render_forecasting_section(df, model_payload)

    with tab_eval:
        render_evaluation_section(phase4a_metrics)

    with tab_backtest:
        render_backtest_section(backtest_metrics)

    with tab_limits:
        render_limitations_section()


if __name__ == "__main__":
    main()
