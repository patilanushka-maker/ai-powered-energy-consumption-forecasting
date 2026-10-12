"""Model evaluation and metrics computation module.

Responsibilities:
- Standardized regression metrics: MAE (kWh), RMSE (kWh), and R² score.
- Missing prediction and evaluation coverage accounting.
- Fair model comparison on individual and common non-missing timestamps.
- JSON metrics report export to outputs/metrics/.
- Actual vs. predicted time-series visualization export to outputs/figures/.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> Dict[str, float]:
    """Compute standard regression metrics on valid (finite) prediction pairs.

    Handles missing/NaN predictions gracefully by filtering to mutual valid pairs.
    If fewer than 1 valid pair exists, metrics return np.nan.

    Metrics computed:
    - MAE: Mean Absolute Error (in kWh)
    - RMSE: Root Mean Squared Error (in kWh)
    - R²: Coefficient of Determination (mathematically valid when variance(y_true) > 0 and n >= 2)

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dict with keys: 'mae', 'rmse', 'r2', 'n_samples'.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if len(yt) != len(yp):
        raise ValueError(
            f"Length mismatch: y_true has {len(yt)} elements, y_pred has {len(yp)} elements."
        )

    # Valid mask where both actual and predicted are non-null and finite
    valid_mask = np.isfinite(yt) & np.isfinite(yp)
    valid_yt = yt[valid_mask]
    valid_yp = yp[valid_mask]
    n_samples = int(len(valid_yt))

    if n_samples == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "n_samples": 0,
        }

    mae = float(mean_absolute_error(valid_yt, valid_yp))
    # Compatible with all scikit-learn versions
    rmse = float(np.sqrt(mean_squared_error(valid_yt, valid_yp)))

    # R2 requires at least 2 samples and non-zero target variance
    if n_samples >= 2 and np.var(valid_yt) > 1e-12:
        r2 = float(r2_score(valid_yt, valid_yp))
    else:
        r2 = float("nan")

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6) if not np.isnan(r2) else float("nan"),
        "n_samples": n_samples,
    }


def align_predictions_to_dataframe(
    df: pd.DataFrame,
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    model_name: Optional[str] = None,
) -> pd.Series:
    """Validate and align prediction Series or array to the evaluation DataFrame index.

    Guarantees:
    - If y_pred is a pd.Series:
      - Reindexes matching index labels to df.index so permutations are aligned correctly.
      - If y_pred's index is completely disjoint from df.index or contains foreign keys,
        raises ValueError rather than silently scoring mismatched timestamps.
    - If y_pred is array-like (np.ndarray, list, tuple):
      - Validates that length exactly matches len(df). If length differs, raises ValueError.
      - Aligns unambiguously to df.index.

    Args:
        df: Evaluation DataFrame with reference index.
        y_pred: Predicted values (Series or array-like).
        model_name: Optional model identifier for error messages.

    Returns:
        pd.Series aligned to df.index with float64 dtype.

    Raises:
        ValueError: If lengths mismatch or Series index does not align with df.index.
    """
    model_desc = f" for model '{model_name}'" if model_name else ""

    if isinstance(y_pred, pd.Series):
        if y_pred.index.equals(df.index):
            return y_pred.astype("float64")

        # Check index compatibility
        df_index_set = set(df.index)
        pred_index_set = set(y_pred.index)
        intersection = df_index_set.intersection(pred_index_set)

        if not intersection:
            raise ValueError(
                f"Prediction Series index{model_desc} is completely disjoint from evaluation DataFrame index. "
                f"DataFrame index: {list(df.index[:5])}..., Prediction index: {list(y_pred.index[:5])}..."
            )

        if not pred_index_set.issubset(df_index_set) and not df_index_set.issubset(pred_index_set):
            raise ValueError(
                f"Prediction Series index{model_desc} contains keys not present in evaluation DataFrame index. "
                f"Unmatched keys: {list(pred_index_set - df_index_set)[:5]}..."
            )

        # Reindex to match df.index ordering precisely
        return y_pred.reindex(df.index).astype("float64")

    # Array-like handling
    arr = np.asarray(y_pred, dtype=np.float64)
    if len(arr) != len(df):
        raise ValueError(
            f"Prediction array length ({len(arr)}){model_desc} does not match "
            f"evaluation DataFrame length ({len(df)})."
        )
    return pd.Series(arr, index=df.index, dtype="float64")


def evaluate_model_predictions(
    df: pd.DataFrame,
    y_pred: Union[np.ndarray, pd.Series],
    model_name: str,
    target_col: str = "energy_kwh",
    timestamp_col: str = "timestamp",
) -> Dict[str, Any]:
    """Evaluate predictions for a single model with missing-prediction accounting.

    Args:
        df: Partition DataFrame containing actual target values.
        y_pred: Predicted values aligned with df.
        model_name: Identifier for the model.
        target_col: Target variable column name (default: 'energy_kwh').
        timestamp_col: Timestamp column name (default: 'timestamp').

    Returns:
        Dict containing model name, row counts, missing prediction counts, coverage, and metrics.
    """
    if target_col not in df.columns:
        raise ValueError(f"Evaluation DataFrame must contain '{target_col}'.")

    total_rows = len(df)
    y_true = pd.to_numeric(df[target_col], errors="coerce").values
    pred_series = align_predictions_to_dataframe(df, y_pred, model_name=model_name)

    # Missing prediction accounting
    valid_actual = np.isfinite(y_true)
    valid_pred = np.isfinite(pred_series.values)
    mutual_valid = valid_actual & valid_pred

    scored_samples = int(mutual_valid.sum())
    missing_predictions = int((~valid_pred).sum())
    coverage_pct = round((scored_samples / total_rows * 100), 2) if total_rows > 0 else 0.0

    metrics = compute_metrics(y_true, pred_series.values)

    return {
        "model_name": model_name,
        "total_rows": total_rows,
        "scored_samples": scored_samples,
        "missing_predictions": missing_predictions,
        "coverage_percentage": coverage_pct,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
    }


def evaluate_and_compare_models(
    df: pd.DataFrame,
    predictions_dict: Dict[str, Union[np.ndarray, pd.Series]],
    target_col: str = "energy_kwh",
    timestamp_col: str = "timestamp",
) -> Dict[str, Any]:
    """Evaluate and compare multiple models on individual and common non-missing timestamps.

    Args:
        df: Partition DataFrame containing actual target values.
        predictions_dict: Dict mapping model name to predicted array/Series.
        target_col: Target column name (default: 'energy_kwh').
        timestamp_col: Timestamp column name (default: 'timestamp').

    Returns:
        Dict with 'individual_evaluations' and 'common_timestamp_evaluations'.
    """
    if target_col not in df.columns:
        raise ValueError(f"Evaluation DataFrame must contain '{target_col}'.")

    y_true = pd.to_numeric(df[target_col], errors="coerce").values
    valid_actual = np.isfinite(y_true)

    individual_evals: Dict[str, Any] = {}
    valid_masks: Dict[str, np.ndarray] = {}
    aligned_predictions: Dict[str, pd.Series] = {}

    for name, preds in predictions_dict.items():
        aligned = align_predictions_to_dataframe(df, preds, model_name=name)
        aligned_predictions[name] = aligned

        eval_res = evaluate_model_predictions(
            df=df,
            y_pred=aligned,
            model_name=name,
            target_col=target_col,
            timestamp_col=timestamp_col,
        )
        individual_evals[name] = eval_res

        pred_arr = aligned.values
        valid_masks[name] = valid_actual & np.isfinite(pred_arr)

    # Identify common timestamps where ALL models produced valid predictions
    common_mask = valid_actual.copy()
    for mask in valid_masks.values():
        common_mask = common_mask & mask

    common_count = int(common_mask.sum())
    common_evals: Dict[str, Any] = {}

    for name, aligned in aligned_predictions.items():
        pred_arr = aligned.values
        if common_count > 0:
            sub_metrics = compute_metrics(y_true[common_mask], pred_arr[common_mask])
        else:
            sub_metrics = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n_samples": 0}

        common_evals[name] = {
            "model_name": name,
            "scored_common_samples": common_count,
            "mae": sub_metrics["mae"],
            "rmse": sub_metrics["rmse"],
            "r2": sub_metrics["r2"],
        }

    return {
        "total_partition_rows": len(df),
        "common_scored_count": common_count,
        "individual_evaluations": individual_evals,
        "common_timestamp_evaluations": common_evals,
    }


def save_metrics_report(
    metrics_data: Dict[str, Any],
    save_path: Union[str, Path],
) -> Path:
    """Save machine-readable metrics evaluation report to JSON.

    Args:
        metrics_data: Dictionary of computed evaluation metrics and metadata.
        save_path: Destination JSON filepath.

    Returns:
        Path of saved JSON file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, default=str)

    logger.info("Saved metrics report to %s", path)
    return path


def plot_predictions(
    timestamps: Union[pd.Series, Sequence[Any]],
    y_true: Union[pd.Series, Sequence[float]],
    predictions_dict: Dict[str, Union[pd.Series, Sequence[float]]],
    save_path: Union[str, Path],
    title: str = "Energy Consumption: Actual vs. Forecast",
    max_plot_points: int = 168,
) -> Path:
    """Generate and save actual vs. predicted consumption plot.

    Args:
        timestamps: Sequence of timestamps.
        y_true: Actual ground truth values.
        predictions_dict: Mapping of model names to predicted values.
        save_path: Destination path for the generated image.
        title: Plot title.
        max_plot_points: Number of points to display (default: 168, 1 week of hourly data).

    Returns:
        Path of saved figure.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ts_arr = pd.to_datetime(list(timestamps))
    yt_arr = np.asarray(y_true, dtype=np.float64)

    # Slice to first max_plot_points for clean visual representation
    n = min(len(ts_arr), max_plot_points)
    plot_ts = ts_arr[:n]
    plot_yt = yt_arr[:n]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    # Plot actual
    ax.plot(plot_ts, plot_yt, label="Actual Consumption", color="#1f77b4", linewidth=2.0)

    # Palette for models
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        p_arr = np.asarray(preds, dtype=np.float64)[:n]
        color = colors[i % len(colors)]
        ax.plot(
            plot_ts,
            p_arr,
            label=f"Predicted ({name})",
            color=color,
            linewidth=1.5,
            linestyle="--" if "Naive" in name or "Persistence" in name else "-",
            alpha=0.85,
        )

    ax.set_title(f"{title} (First {n} Hours)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("Energy Consumption (kWh)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.xticks(rotation=25)
    plt.tight_layout()

    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved forecast comparison plot to %s", path)
    return path


def evaluate_and_export_results(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    output_dir: Path = Path("outputs"),
) -> Dict[str, float]:
    """Evaluate test split predictions and export results to output directories.

    Args:
        test_df: Test split DataFrame with actual target values.
        predictions: Model predictions array.
        model_name: Identifier for model being evaluated.
        output_dir: Directory where figures and metrics are stored.

    Returns:
        Dict of computed evaluation metrics (mae, rmse, r2, n_samples).
    """
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    eval_result = evaluate_model_predictions(
        df=test_df,
        y_pred=predictions,
        model_name=model_name,
    )

    metrics_file = metrics_dir / f"{model_name.lower().replace(' ', '_')}_metrics.json"
    save_metrics_report(eval_result, metrics_file)

    if "timestamp" in test_df.columns:
        figure_file = figures_dir / f"{model_name.lower().replace(' ', '_')}_vs_actual.png"
        plot_predictions(
            timestamps=test_df["timestamp"],
            y_true=test_df["energy_kwh"],
            predictions_dict={model_name: predictions},
            save_path=figure_file,
            title=f"{model_name} Forecast vs Actual",
        )

    return {
        "mae": eval_result["mae"],
        "rmse": eval_result["rmse"],
        "r2": eval_result["r2"],
        "n_samples": eval_result["scored_samples"],
    }
