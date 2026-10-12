"""Main pipeline execution script for AI Energy Consumption Forecasting.

Orchestrates pipeline stages via command line arguments:
- preprocess: Ingests raw UCI dataset, cleans timestamps, resolves duplicates,
  and resamples to hourly energy consumption (kWh) with coverage gating.
- train: Chronological splitting, feature engineering, baseline models (Persistence,
  Daily Seasonal Naive), Random Forest training, and evaluation.
- evaluate: Evaluation and comparative benchmark reporting.
- forecast: Multi-step forecasting inference (Phase 4B/5).
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_loader import load_raw_data
from src.evaluate import (
    evaluate_and_compare_models,
    plot_predictions,
    save_metrics_report,
)
from src.feature_engineering import (
    create_lag_and_rolling_features,
    extract_temporal_features,
)
from src.forecast import generate_forecast
from src.preprocessing import clean_and_resample_hourly
from src.train import (
    STANDARD_PREDICTOR_COLS,
    chronological_split,
    filter_eligible_rows,
    predict_daily_seasonal_naive,
    predict_persistence,
    select_predictor_features,
    train_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the energy forecasting pipeline."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Energy Consumption Forecasting Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["preprocess", "train", "evaluate", "forecast", "all"],
        default="preprocess",
        help="Pipeline stage to execute (preprocess, train, evaluate, forecast, all).",
    )
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=Path("data/raw/household_power_consumption.txt"),
        help="Path to the raw UCI dataset text file (default: data/raw/household_power_consumption.txt).",
    )
    parser.add_argument(
        "--processed-data-path",
        type=Path,
        default=Path("data/processed/hourly_energy.csv"),
        help="Destination path for the processed hourly CSV (default: data/processed/hourly_energy.csv).",
    )
    parser.add_argument(
        "--min-hourly-coverage",
        type=float,
        default=0.90,
        help="Minimum required fraction of valid minute observations per hour (default: 0.90).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest"],
        help="Model architecture to train (default: random_forest).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Chronological train split ratio (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Chronological validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees for Random Forest (default: 100).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--model-save-path",
        type=Path,
        default=Path("models/random_forest_model.joblib"),
        help="Path to save trained model artifact (default: models/random_forest_model.joblib).",
    )
    parser.add_argument(
        "--metrics-save-path",
        type=Path,
        default=Path("outputs/metrics/phase4a_metrics.json"),
        help="Path to save evaluation metrics report (default: outputs/metrics/phase4a_metrics.json).",
    )
    parser.add_argument(
        "--figure-save-path",
        type=Path,
        default=Path("outputs/figures/phase4a_actual_vs_predicted.png"),
        help="Path to save test prediction figure (default: outputs/figures/phase4a_actual_vs_predicted.png).",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=24,
        help="Forecast horizon in hours (1 to 168 inclusive, default: 24).",
    )
    parser.add_argument(
        "--forecast-output-path",
        type=Path,
        default=Path("outputs/forecasts/forecast_24h.csv"),
        help="Destination path for forecast predictions CSV (default: outputs/forecasts/forecast_24h.csv).",
    )
    return parser.parse_args()


def run_preprocess_stage(
    raw_path: Path,
    processed_path: Path,
    min_hourly_coverage: float,
) -> None:
    """Execute the data loading and preprocessing pipeline stage."""
    print("=" * 65)
    print("STAGE: Data Loading & Preprocessing (Phase 2)")
    print(f"Raw source path       : {raw_path}")
    print(f"Processed target path : {processed_path}")
    print(f"Min hourly coverage   : {min_hourly_coverage:.1%}")
    print("=" * 65)

    if not raw_path.exists():
        print(
            f"ERROR: Raw dataset not found at '{raw_path}'.\n"
            "Please download 'household_power_consumption.zip' from the UCI Machine Learning Repository:\n"
            "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption\n"
            f"and extract 'household_power_consumption.txt' into '{raw_path.parent}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Step 1: Load raw data
        print("\n[1/3] Loading raw dataset...")
        raw_df = load_raw_data(raw_path)
        loader_audit = raw_df.attrs.get("data_loader_audit", {})
        print(
            f"      Loaded {len(raw_df):,} raw minute observations "
            f"({loader_audit.get('missing_power_readings', 0):,} missing/invalid power readings)."
        )

        # Step 2: Clean, resolve duplicates, and resample hourly
        print("\n[2/3] Cleaning, resolving duplicates, and computing hourly energy...")
        hourly_df, audit = clean_and_resample_hourly(
            raw_df,
            min_hourly_coverage=min_hourly_coverage,
            return_audit=True,
        )

        if len(hourly_df) == 0:
            print(
                "ERROR: Preprocessing yielded 0 qualifying hourly intervals. "
                "Output file will NOT be created.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Step 3: Save processed dataset safely
        print(f"\n[3/3] Saving processed hourly dataset to '{processed_path}'...")
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        hourly_df.to_csv(processed_path, index=False)

        if not processed_path.exists() or processed_path.stat().st_size == 0:
            print(
                f"ERROR: Failed to verify written processed output at '{processed_path}'.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"      Successfully saved {len(hourly_df):,} hourly rows.")

        # Print audit summary
        print("\n" + "-" * 65)
        print("PREPROCESSING AUDIT SUMMARY")
        print("-" * 65)
        print(f"Original minute rows         : {audit.original_rows:,}")
        print(f"Invalid timestamps           : {audit.invalid_timestamps:,}")
        print(f"Invalid power readings       : {audit.invalid_power_readings:,}")
        print(f"Duplicate timestamps         : {audit.duplicate_timestamps:,}")
        print(f"Conflicting duplicates       : {audit.conflicting_duplicates:,}")
        print(f"Accepted hourly intervals    : {audit.accepted_intervals:,}")
        print(f"Excluded hourly intervals    : {audit.excluded_intervals:,}")
        print(f"Date range                   : {audit.earliest_timestamp} -> {audit.latest_timestamp}")
        print(f"Output file location         : {processed_path.resolve()}")
        print("-" * 65)
        print("Preprocessing stage completed successfully.")

    except Exception as exc:
        print(f"ERROR: Preprocessing pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


def run_train_stage(
    processed_path: Path,
    raw_path: Path,
    min_hourly_coverage: float,
    train_ratio: float,
    val_ratio: float,
    model_type: str,
    n_estimators: int,
    random_state: int,
    model_save_path: Path,
    metrics_save_path: Path,
    figure_save_path: Path,
) -> None:
    """Execute chronological splitting, feature engineering, baselines, and model training."""
    print("=" * 65)
    print("STAGE: Model Training & Evaluation (Phase 4A)")
    print(f"Processed data path  : {processed_path}")
    print(f"Model architecture   : {model_type}")
    print(f"Split proportions    : train={train_ratio:.0%}, val={val_ratio:.0%}, test={1 - train_ratio - val_ratio:.0%}")
    print(f"Random seed          : {random_state}")
    print("=" * 65)

    # 1. Check data availability
    if not processed_path.exists():
        if raw_path.exists():
            print(f"Processed file '{processed_path}' not found, but raw file exists. Preprocessing first...")
            run_preprocess_stage(raw_path, processed_path, min_hourly_coverage)
        else:
            print(
                f"\nERROR: Required dataset is missing.\n"
                f"Neither processed data at '{processed_path}'\n"
                f"nor raw data at '{raw_path}' was found.\n\n"
                "Real-world model training and evaluation remain blocked.\n\n"
                "To run training on the real dataset:\n"
                "1. Download 'household_power_consumption.zip' from the UCI Machine Learning Repository:\n"
                "   https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption\n"
                f"2. Extract 'household_power_consumption.txt' into '{raw_path.parent}'.\n"
                "3. Run: python main.py --stage preprocess\n"
                "4. Run: python main.py --stage train\n",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        # 2. Load processed dataset
        print("\n[1/6] Loading processed hourly dataset...")
        df = pd.read_csv(processed_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"      Loaded {len(df):,} hourly observations ({df['timestamp'].min()} -> {df['timestamp'].max()}).")

        # 3. Feature engineering
        print("\n[2/6] Generating temporal and lag features...")
        df_temporal = extract_temporal_features(df)
        df_features = create_lag_and_rolling_features(df_temporal)
        print(f"      Features generated: {list(df_features.columns)}")

        # 4. Chronological splitting
        print("\n[3/6] Partitioning dataset chronologically (no shuffling)...")
        train_df, val_df, test_df, split_summary = chronological_split(
            df_features,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            return_summary=True,
        )

        print(f"      Train: {split_summary.train_rows:,} rows ({split_summary.train_ratio_achieved:.1%}) [{split_summary.train_start} -> {split_summary.train_end}]")
        print(f"      Val  : {split_summary.val_rows:,} rows ({split_summary.val_ratio_achieved:.1%}) [{split_summary.val_start} -> {split_summary.val_end}]")
        print(f"      Test : {split_summary.test_rows:,} rows ({split_summary.test_ratio_achieved:.1%}) [{split_summary.test_start} -> {split_summary.test_end}]")

        # 5. Predictor selection and eligibility filtering
        feature_cols = select_predictor_features(df_features)
        print(f"\n[4/6] Predictor features ({len(feature_cols)}): {feature_cols}")

        train_eligible, train_audit = filter_eligible_rows(train_df, feature_cols)
        val_eligible, val_audit = filter_eligible_rows(val_df, feature_cols)
        test_eligible, test_audit = filter_eligible_rows(test_df, feature_cols)

        print(f"      Eligible rows: train={len(train_eligible):,}/{len(train_df):,} (excluded {train_audit['excluded_rows']:,}), "
              f"val={len(val_eligible):,}/{len(val_df):,} (excluded {val_audit['excluded_rows']:,}), "
              f"test={len(test_eligible):,}/{len(test_df):,} (excluded {test_audit['excluded_rows']:,})")

        # 6. Baseline and ML model predictions
        print("\n[5/6] Fitting model and generating baseline predictions...")

        # Baseline A: Persistence (1-hour lookback)
        val_persist = predict_persistence(val_df, history_df=train_df)
        test_persist = predict_persistence(test_df, history_df=pd.concat([train_df, val_df]))

        # Baseline B: Daily Seasonal Naive (24-hour lookback)
        val_seasonal = predict_daily_seasonal_naive(val_df, history_df=train_df)
        test_seasonal = predict_daily_seasonal_naive(test_df, history_df=pd.concat([train_df, val_df]))

        # ML Model: Random Forest
        rf_model = train_model(
            train_df=train_eligible,
            model_type=model_type,
            feature_cols=feature_cols,
            save_path=model_save_path,
            random_state=random_state,
            n_estimators=n_estimators,
        )

        # Generate ML predictions
        # For rows with missing features, predict NaN
        val_rf_preds = pd.Series(np.nan, index=val_df.index, dtype="float64")
        if len(val_eligible) > 0:
            val_rf_preds.loc[val_eligible.index] = rf_model.predict(val_eligible[feature_cols])

        test_rf_preds = pd.Series(np.nan, index=test_df.index, dtype="float64")
        if len(test_eligible) > 0:
            test_rf_preds.loc[test_eligible.index] = rf_model.predict(test_eligible[feature_cols])

        # 7. Model evaluation
        print("\n[6/6] Evaluating models on validation and test partitions...")

        val_comparisons = evaluate_and_compare_models(
            df=val_df,
            predictions_dict={
                "Persistence (1h)": val_persist,
                "Seasonal Naive (24h)": val_seasonal,
                "Random Forest": val_rf_preds,
            },
        )

        test_comparisons = evaluate_and_compare_models(
            df=test_df,
            predictions_dict={
                "Persistence (1h)": test_persist,
                "Seasonal Naive (24h)": test_seasonal,
                "Random Forest": test_rf_preds,
            },
        )

        # Save metrics report
        metrics_report = {
            "experiment_phase": "Phase 4A",
            "model_type": model_type,
            "random_state": random_state,
            "feature_columns": feature_cols,
            "target_column": "energy_kwh",
            "split_summary": split_summary.to_dict(),
            "eligibility_audits": {
                "train": train_audit,
                "validation": val_audit,
                "test": test_audit,
            },
            "validation_evaluation": val_comparisons,
            "test_evaluation": test_comparisons,
        }

        save_metrics_report(metrics_report, metrics_save_path)

        # Plot actual vs predicted on test partition (first 168 hours = 1 week)
        plot_predictions(
            timestamps=test_df["timestamp"],
            y_true=test_df["energy_kwh"],
            predictions_dict={
                "Persistence": test_persist,
                "Seasonal Naive": test_seasonal,
                "Random Forest": test_rf_preds,
            },
            save_path=figure_save_path,
            title="Phase 4A: Test Split Forecast Benchmark",
            max_plot_points=168,
        )

        # Print summary results table
        print("\n" + "=" * 65)
        print("PHASE 4A EVALUATION SUMMARY (COMMON TEST TIMESTAMPS)")
        print("=" * 65)
        print(f"{'Model':<24} | {'MAE (kWh)':<10} | {'RMSE (kWh)':<10} | {'R²':<8} | {'Scored':<8}")
        print("-" * 65)
        for name, res in test_comparisons["common_timestamp_evaluations"].items():
            mae_str = f"{res['mae']:.4f}" if pd.notna(res['mae']) else "N/A"
            rmse_str = f"{res['rmse']:.4f}" if pd.notna(res['rmse']) else "N/A"
            r2_str = f"{res['r2']:.4f}" if pd.notna(res['r2']) else "N/A"
            scored_str = f"{res['scored_common_samples']:,}"
            print(f"{name:<24} | {mae_str:<10} | {rmse_str:<10} | {r2_str:<8} | {scored_str:<8}")
        print("=" * 65)
        print(f"Artifacts saved:")
        print(f"  Model   : {model_save_path.resolve()}")
        print(f"  Metrics : {metrics_save_path.resolve()}")
        print(f"  Plot    : {figure_save_path.resolve()}")
        print("Phase 4A training and evaluation pipeline completed successfully.")

    except Exception as exc:
        print(f"ERROR: Model training pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


def run_forecast_stage(
    model_path: Path,
    processed_path: Path,
    horizon_hours: int,
    forecast_output_path: Path,
) -> None:
    """Execute multi-step recursive forecasting inference (Phase 4B)."""
    print("=" * 65)
    print("STAGE: Multi-Step Forecasting Inference (Phase 4B)")
    print(f"Model artifact path   : {model_path}")
    print(f"Historical data path  : {processed_path}")
    print(f"Forecast horizon      : {horizon_hours} hours")
    print(f"Destination CSV path  : {forecast_output_path}")
    print("=" * 65)

    if not model_path.exists():
        print(
            f"ERROR: Model artifact not found at '{model_path}'.\n"
            "Please run model training first: python main.py --stage train",
            file=sys.stderr,
        )
        sys.exit(1)

    if not processed_path.exists():
        print(
            f"ERROR: Processed historical data not found at '{processed_path}'.\n"
            "Please run preprocessing first: python main.py --stage preprocess",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        forecast_df = generate_forecast(
            model_or_path=model_path,
            history=processed_path,
            horizon_hours=horizon_hours,
            output_path=forecast_output_path,
        )

        print("\nFORECAST GENERATION SUMMARY")
        print("-" * 65)
        print(f"Forecast origin       : {forecast_df.attrs.get('forecast_origin')}")
        print(f"History window start  : {forecast_df.attrs.get('history_start')}")
        print(f"History window end    : {forecast_df.attrs.get('history_end')}")
        print(f"Horizon hours         : {forecast_df.attrs.get('horizon_hours')}")
        print(f"Model type            : {forecast_df.attrs.get('model_type')}")
        print(f"Output rows generated : {len(forecast_df):,}")
        print(f"Output saved to       : {forecast_output_path.resolve()}")
        print(f"First step prediction : {forecast_df['timestamp'].iloc[0]} -> {forecast_df['forecast_energy_kwh'].iloc[0]:.4f} kWh")
        print(f"Final step prediction : {forecast_df['timestamp'].iloc[-1]} -> {forecast_df['forecast_energy_kwh'].iloc[-1]:.4f} kWh")
        print("-" * 65)
        print("Forecast stage completed successfully.")

    except Exception as exc:
        print(f"ERROR: Forecasting pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main pipeline entry point."""
    args = parse_args()

    if args.stage == "preprocess":
        run_preprocess_stage(
            raw_path=args.raw_data_path,
            processed_path=args.processed_data_path,
            min_hourly_coverage=args.min_hourly_coverage,
        )
    elif args.stage == "train":
        run_train_stage(
            processed_path=args.processed_data_path,
            raw_path=args.raw_data_path,
            min_hourly_coverage=args.min_hourly_coverage,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            model_save_path=args.model_save_path,
            metrics_save_path=args.metrics_save_path,
            figure_save_path=args.figure_save_path,
        )
    elif args.stage == "evaluate":
        # Evaluation is integrated with model training pipeline
        run_train_stage(
            processed_path=args.processed_data_path,
            raw_path=args.raw_data_path,
            min_hourly_coverage=args.min_hourly_coverage,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            model_save_path=args.model_save_path,
            metrics_save_path=args.metrics_save_path,
            figure_save_path=args.figure_save_path,
        )
    elif args.stage == "forecast":
        run_forecast_stage(
            model_path=args.model_save_path,
            processed_path=args.processed_data_path,
            horizon_hours=args.horizon_hours,
            forecast_output_path=args.forecast_output_path,
        )
    elif args.stage == "all":
        # Preprocess then Train
        run_preprocess_stage(
            raw_path=args.raw_data_path,
            processed_path=args.processed_data_path,
            min_hourly_coverage=args.min_hourly_coverage,
        )
        run_train_stage(
            processed_path=args.processed_data_path,
            raw_path=args.raw_data_path,
            min_hourly_coverage=args.min_hourly_coverage,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            model_save_path=args.model_save_path,
            metrics_save_path=args.metrics_save_path,
            figure_save_path=args.figure_save_path,
        )


if __name__ == "__main__":
    main()
