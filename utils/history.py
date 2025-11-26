"""
Training history management using SQLite database.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import plotly.graph_objects as go

DB_PATH = Path("data/training_history.db")

def init_database():
    """Initialize the training history database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            data_yml TEXT,
            epochs INTEGER,
            learning_rate REAL,
            batch_size INTEGER,
            optimizer TEXT,
            image_size INTEGER,
            patience INTEGER,
            weight_decay REAL,
            momentum REAL,
            final_loss REAL,
            final_accuracy REAL,
            best_loss REAL,
            best_accuracy REAL,
            weights_path TEXT
        )
    """)
    
    conn.commit()
    conn.close()


def save_training_run(config: dict, metrics: dict, weights_path: str = None, results_path: str = None) -> str:
    """
    Save a training run to the database.
    
    Args:
        config: Training configuration dictionary (user-provided values)
        metrics: Training metrics dictionary
        weights_path: Path to the best.pt weights file (optional)
        results_path: Path to training results directory (to extract args.yaml)
    
    Returns:
        Run ID (timestamp-based)
    """
    init_database()
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat()
    
    # Extract actual config from args.yaml to fill in missing values
    complete_config = config.copy()
    if results_path:
        from utils.yolo_metrics import extract_training_args
        actual_config = extract_training_args(Path(results_path))
        if actual_config:
            # Merge: use user-provided values if available, otherwise use actual (default) values
            for key in ["epochs", "batch_size", "image_size", "learning_rate", 
                        "momentum", "weight_decay", "patience", "optimizer"]:
                if key not in complete_config or complete_config.get(key) is None:
                    if key in actual_config and actual_config[key] is not None:
                        complete_config[key] = actual_config[key]
    
    # Use provided weights path or try to find it from results_path
    if weights_path is None:
        # Try to get from metrics if available
        weights_path = metrics.get("weights_path")
        if weights_path is None:
            # Fallback: try to find latest run
            from utils.yolo_metrics import find_latest_yolo_run
            latest_run = find_latest_yolo_run("runs/segment")
            if latest_run:
                best_weights = latest_run / "weights" / "best.pt"
                if not best_weights.exists():
                    best_weights = latest_run / "best.pt"
                if best_weights.exists():
                    weights_path = str(best_weights)
            if weights_path is None:
                weights_path = "runs/segment/train/weights/best.pt"  # Default fallback
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO training_runs (
            run_id, timestamp, data_yml, epochs, learning_rate, batch_size,
            optimizer, image_size, patience, weight_decay, momentum,
            final_loss, final_accuracy, best_loss, best_accuracy, weights_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        timestamp,
        complete_config.get("data_yml"),
        complete_config.get("epochs"),
        complete_config.get("learning_rate"),
        complete_config.get("batch_size"),
        complete_config.get("optimizer"),
        complete_config.get("image_size"),
        complete_config.get("patience"),
        complete_config.get("weight_decay"),
        complete_config.get("momentum"),
        metrics.get("final_loss"),
        metrics.get("final_accuracy"),
        metrics.get("best_loss"),
        metrics.get("best_accuracy"),
        weights_path
    ))
    
    conn.commit()
    conn.close()
    
    return run_id


def get_training_history() -> pd.DataFrame:
    """
    Retrieve all training runs from the database.
    
    Returns:
        DataFrame with training history, or None if no data
    """
    init_database()
    
    if not DB_PATH.exists():
        return None
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        df = pd.read_sql_query("SELECT * FROM training_runs ORDER BY timestamp DESC", conn)
        conn.close()
        
        if len(df) == 0:
            return None
        
        # Format timestamp for display
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df
    except Exception as e:
        conn.close()
        return None


def compare_runs(run_id1: str, run_id2: str) -> dict:
    """
    Compare two training runs and generate comparison charts.
    
    Args:
        run_id1: First run ID
        run_id2: Second run ID
    
    Returns:
        Dictionary with comparison metrics and charts
    """
    init_database()
    
    if not DB_PATH.exists():
        return None
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Get both runs
        df = pd.read_sql_query(
            "SELECT * FROM training_runs WHERE run_id IN (?, ?)",
            conn,
            params=(run_id1, run_id2)
        )
        conn.close()
        
        if len(df) != 2:
            return None
        
        run1 = df[df['run_id'] == run_id1].iloc[0]
        run2 = df[df['run_id'] == run_id2].iloc[0]
        
        # Compare metrics
        metrics = {
            "Final Loss Diff": abs(run1['final_loss'] - run2['final_loss']),
            "Final Accuracy Diff": abs(run1['final_accuracy'] - run2['final_accuracy']),
            "Best Loss Diff": abs(run1['best_loss'] - run2['best_loss']),
            "Best Accuracy Diff": abs(run1['best_accuracy'] - run2['best_accuracy'])
        }
        
        # Create comparison charts
        charts = {}
        
        # Loss comparison
        fig_loss = go.Figure()
        fig_loss.add_bar(
            x=[run_id1, run_id2],
            y=[run1['final_loss'], run2['final_loss']],
            name="Final Loss",
            marker_color=["#4CAF50", "#FF9800"]
        )
        fig_loss.update_layout(
            title="Final Loss Comparison",
            yaxis_title="Loss",
            template="plotly_white"
        )
        charts["loss_comparison"] = fig_loss
        
        # Accuracy comparison
        fig_acc = go.Figure()
        fig_acc.add_bar(
            x=[run_id1, run_id2],
            y=[run1['final_accuracy'], run2['final_accuracy']],
            name="Final Accuracy",
            marker_color=["#2196F3", "#9C27B0"]
        )
        fig_acc.update_layout(
            title="Final Accuracy Comparison",
            yaxis_title="Accuracy",
            template="plotly_white"
        )
        charts["accuracy_comparison"] = fig_acc
        
        return {
            "metrics": metrics,
            "charts": charts,
            "run1": run1.to_dict(),
            "run2": run2.to_dict()
        }
    except Exception as e:
        conn.close()
        return None

