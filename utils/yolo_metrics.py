"""
Utility functions to read and parse YOLO training metrics from results files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import glob

def find_latest_yolo_run(base_path: str = "runs/segment") -> Optional[Path]:
    """
    Find the latest YOLO training run directory.
    YOLO creates numbered directories: train, train2, train3, etc.
    This function finds the most recently created one.
    
    Args:
        base_path: Base path to YOLO training runs (default: "runs/segment")
    
    Returns:
        Path to latest experiment directory or None
    """
    base = Path(base_path)
    if not base.exists():
        return None
    
    # Check if this is a direct experiment directory (e.g., runs/segment/train)
    if base.name.startswith("train") and (base / "results.csv").exists():
        return base
    
    # Find all experiment directories that start with "train" (train, train2, train3, etc.)
    # Only include directories that have results.csv (completed training runs)
    train_dirs = [
        d for d in base.iterdir() 
        if d.is_dir() and d.name.startswith("train") and (d / "results.csv").exists()
    ]
    
    if not train_dirs:
        return None
    
    # Sort by modification time (most recent first), with name as secondary sort
    # This ensures we get the truly latest training run
    # Modification time is the most reliable indicator of when training completed
    train_dirs_sorted = sorted(
        train_dirs,
        key=lambda x: (x.stat().st_mtime, x.name),
        reverse=True
    )
    
    # Return the most recently modified directory
    return train_dirs_sorted[0]


def load_yolo_results(exp_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load YOLO training results from results.csv file.
    
    Args:
        exp_path: Path to experiment directory. If None, finds latest.
    
    Returns:
        DataFrame with training metrics or None if not found
    """
    if exp_path is None:
        exp_path = find_latest_yolo_run()
    
    if exp_path is None:
        return None
    
    results_file = exp_path / "results.csv"
    
    if not results_file.exists():
        return None
    
    try:
        df = pd.read_csv(results_file)
        return df
    except Exception as e:
        print(f"Error reading results.csv: {e}")
        return None


def extract_yolo_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract final metrics from YOLO results DataFrame.
    
    Args:
        df: DataFrame with YOLO training results
    
    Returns:
        Dictionary with extracted metrics
    """
    if df is None or len(df) == 0:
        return {}
    
    # Get the last row (final epoch)
    last_row = df.iloc[-1]
    
    metrics = {}
    
    # Map YOLO column names to our metric names
    # YOLO typically uses these column names (may vary by version)
    column_mapping = {
        # Precision and Recall
        "metrics/precision(M)": "precision",
        "metrics/recall(M)": "recall",
        "metrics/mAP50(M)": "map50",
        "metrics/mAP50-95(M)": "map50_95",
        
        # Losses
        "train/box_loss": "train_box_loss",
        "train/seg_loss": "train_seg_loss",
        "train/cls_loss": "train_cls_loss",
        "val/box_loss": "val_box_loss",
        "val/seg_loss": "val_seg_loss",
        "val/cls_loss": "val_cls_loss",
    }
    
    # Try to find columns (YOLO column names may vary)
    for yolo_col, metric_name in column_mapping.items():
        # Check for exact match or partial match
        matching_cols = [col for col in df.columns if yolo_col.lower() in col.lower()]
        if matching_cols:
            metrics[metric_name] = float(last_row[matching_cols[0]])
    
    # Calculate IoU if we have segmentation metrics
    # For segmentation, we can approximate IoU from mAP or use specific metrics if available
    
    # Get mAP (use mAP50-95 as primary mAP metric)
    if "map50_95" in metrics:
        metrics["map50_95"] = metrics["map50_95"]
        
    if "map50" in metrics:
        metrics["map50"] = metrics["map50"]
    
    # Calculate accuracy from precision and recall if available
    if "precision" in metrics and "recall" in metrics:
        # F1 score as a proxy for accuracy
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["accuracy"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    
    # Get best metrics (minimum loss, maximum mAP) and final loss (last epoch)
    val_loss_cols = [col for col in df.columns if "val" in col.lower() and "loss" in col.lower()]
    if val_loss_cols:
        total_val_loss = df[val_loss_cols].sum(axis=1)
        metrics["best_loss"] = float(total_val_loss.min())
        metrics["final_loss"] = float(total_val_loss.iloc[-1])  # Last epoch's validation loss
    
    if "map" in metrics:
        map_col = [col for col in df.columns if "map50-95" in col.lower() or "map50" in col.lower()]
        if map_col:
            metrics["best_map"] = float(df[map_col[0]].max())
    
    return metrics


def get_training_data_from_yolo(df: pd.DataFrame) -> Dict[str, list]:
    """
    Extract training data for plotting from YOLO results.
    
    Args:
        df: DataFrame with YOLO training results
    
    Returns:
        Dictionary with epoch, loss, and accuracy data
    """
    if df is None or len(df) == 0:
        return {"epoch": [], "loss": [], "accuracy": [], "map": []}
    
    # Handle epoch column - YOLO results usually have epoch starting from 0 or 1
    if "epoch" in df.columns:
        data = {"epoch": (df["epoch"] + 1).tolist()}  # Make epochs 1-indexed
    else:
        # Use index + 1 as epoch number
        data = {"epoch": [i + 1 for i in range(len(df))]}
    
    # Extract loss data - YOLO typically has multiple loss components
    train_loss_cols = [col for col in df.columns if "train" in col.lower() and "loss" in col.lower()]
    val_loss_cols = [col for col in df.columns if "val" in col.lower() and "loss" in col.lower()]
    
    if train_loss_cols:
        # Sum all training loss components
        data["train_loss"] = df[train_loss_cols].sum(axis=1).tolist()
    if val_loss_cols:
        # Sum all validation loss components
        data["val_loss"] = df[val_loss_cols].sum(axis=1).tolist()
    
    # Extract mAP data
    map_cols = [col for col in df.columns if "map" in col.lower()]
    if map_cols:
        data["map"] = df[map_cols[0]].tolist()
    
    # Calculate accuracy from precision and recall if available
    # For segmentation, prioritize mask metrics (M) over box metrics (B)
    precision_cols = [col for col in df.columns if "precision" in col.lower()]
    recall_cols = [col for col in df.columns if "recall" in col.lower()]
    
    # Prefer segmentation/mask metrics (M) over box metrics (B)
    precision_col = None
    recall_col = None
    
    # Look for mask/segmentation metrics first
    for col in precision_cols:
        if "(m)" in col.lower() or "mask" in col.lower() or "seg" in col.lower():
            precision_col = col
            break
    
    for col in recall_cols:
        if "(m)" in col.lower() or "mask" in col.lower() or "seg" in col.lower():
            recall_col = col
            break
    
    # Fallback to first available if mask metrics not found
    if precision_col is None and precision_cols:
        precision_col = precision_cols[0]
    if recall_col is None and recall_cols:
        recall_col = recall_cols[0]
    
    if precision_col and recall_col:
        precision = df[precision_col].values
        recall = df[recall_col].values
        # F1 score as accuracy proxy
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        data["accuracy"] = f1.tolist()
        data["train_accuracy"] = f1.tolist()
        data["val_accuracy"] = f1.tolist()
    
    return data


def get_yolo_metrics() -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame]]:
    """
    Get latest YOLO metrics and results DataFrame.
    
    Returns:
        Tuple of (metrics_dict, results_dataframe)
    """
    # Look in runs/segment first, then fallback to other locations
    latest_run = find_latest_yolo_run("runs/segment")
    if latest_run is None:
        latest_run = find_latest_yolo_run("runs/train")
    if latest_run is None:
        latest_run = find_latest_yolo_run()
    
    df = load_yolo_results(latest_run)
    if df is None:
        return None, None
    
    metrics = extract_yolo_metrics(df)
    return metrics, df


def extract_training_args(exp_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Extract actual training hyperparameters from YOLO's args.yaml file.
    This includes default values that YOLO used.
    
    Args:
        exp_path: Path to experiment directory. If None, finds latest.
    
    Returns:
        Dictionary with actual hyperparameters used, or None if not found
    """
    if exp_path is None:
        exp_path = find_latest_yolo_run()
    
    if exp_path is None:
        return None
    
    args_file = exp_path / "args.yaml"
    
    if not args_file.exists():
        return None
    
    try:
        import yaml
        with open(args_file, 'r') as f:
            args = yaml.safe_load(f)
        
        # Map YOLO args to our config format
        actual_config = {
            "epochs": args.get("epochs"),
            "batch_size": args.get("batch"),
            "image_size": args.get("imgsz"),
            "learning_rate": args.get("lr0"),
            "momentum": args.get("momentum"),
            "weight_decay": args.get("weight_decay"),
            "patience": args.get("patience"),
        }
        
        # Handle optimizer - normalize the value
        optimizer_arg = args.get("optimizer", "auto")
        if optimizer_arg and optimizer_arg != "auto":
            optimizer_lower = str(optimizer_arg).lower()
            if optimizer_lower in ['adam', 'adamw', 'sgd', 'rmsprop']:
                if optimizer_lower == 'adamw':
                    actual_config["optimizer"] = 'AdamW'
                elif optimizer_lower == 'adam':
                    actual_config["optimizer"] = 'Adam'
                elif optimizer_lower == 'sgd':
                    actual_config["optimizer"] = 'SGD'
                elif optimizer_lower == 'rmsprop':
                    actual_config["optimizer"] = 'RMSprop'
            else:
                actual_config["optimizer"] = str(optimizer_arg)
        # If optimizer is "auto", we'll leave it as None (user didn't specify)
        
        return actual_config
    except Exception as e:
        print(f"Error reading args.yaml: {e}")
        return None

