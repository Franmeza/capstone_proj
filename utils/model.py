"""
Model utility functions for YOLO inference and training.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import random
from ultralytics import YOLO
import pandas as pd
from typing import Callable, Optional, Dict

def predict_image_or_video(input_path: str, file_type: str = "image") -> dict:
    """
    Placeholder function for YOLO inference on images or videos.
    
    Args:
        input_path: Path to input image or video file
        file_type: "image" or "video"
    
    Returns:
        Dictionary containing:
        - output_path: Path to processed output file
        - corroded_area_pct: Percentage of image/video with corrosion
        - confidence: Average confidence score
        - num_regions: Number of detected corrosion regions
        - severity: Severity level (Low/Medium/High)
    """
    
    output_dir = Path("runs/predict")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if file_type == "image":
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Placeholder: Create segmentation overlay
        # In real implementation, this would use YOLO model
        overlay = create_dummy_segmentation_overlay(img)
        
        # Save result
        output_path = output_dir / f"processed_{Path(input_path).name}"
        cv2.imwrite(str(output_path), overlay)
        
        # Calculate dummy statistics
        h, w = img.shape[:2]
        corroded_area_pct = random.uniform(10, 50)
        confidence = random.uniform(0.85, 0.98)
        num_regions = random.randint(1, 5)
        
        severity = "Low" if corroded_area_pct < 20 else "Medium" if corroded_area_pct < 35 else "High"
        
    else:  # video
        # Load video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not load video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        output_path = output_dir / f"processed_{Path(input_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_corroded = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Placeholder: Process frame with segmentation
            processed_frame = create_dummy_segmentation_overlay(frame)
            out.write(processed_frame)
            
            frame_count += 1
            total_corroded += random.uniform(10, 50)
        
        cap.release()
        out.release()
        
        # Calculate statistics
        corroded_area_pct = total_corroded / frame_count if frame_count > 0 else 0
        confidence = random.uniform(0.85, 0.98)
        num_regions = random.randint(1, 5)
        severity = "Low" if corroded_area_pct < 20 else "Medium" if corroded_area_pct < 35 else "High"
    
    return {
        "output_path": str(output_path),
        "corroded_area_pct": corroded_area_pct,
        "confidence": confidence,
        "num_regions": num_regions,
        "severity": severity
    }


def create_dummy_segmentation_overlay(img: np.ndarray) -> np.ndarray:
    """
    Create a dummy segmentation overlay for visualization.
    In real implementation, this would use YOLO model predictions.
    
    Args:
        img: Input image as numpy array
    
    Returns:
        Image with segmentation overlay
    """
    overlay = img.copy()
    h, w = img.shape[:2]
    
    # Create random segmentation mask (dummy)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Add random corrosion regions
    num_regions = random.randint(1, 5)
    for _ in range(num_regions):
        center_x = random.randint(w//4, 3*w//4)
        center_y = random.randint(h//4, 3*h//4)
        radius = random.randint(20, min(w, h)//4)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Apply overlay with transparency
    overlay[mask > 0] = [0, 0, 255]  # Red for corrosion
    result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    return result


def train_model(config: dict, progress_callback: Optional[Callable] = None) -> dict:
    """
    Train YOLO11n-seg model for segmentation.
    
    Args:
        config: Training configuration dictionary containing:
            - data_yml: Path to dataset YAML
            - epochs: Number of epochs
            - learning_rate: Learning rate
            - batch_size: Batch size
            - optimizer: Optimizer name
            - image_size: Input image size
            - patience: Early stopping patience
            - weight_decay: Weight decay
            - momentum: Momentum
        progress_callback: Optional callback function for progress updates
            Should accept (epoch, total_epochs, metrics_dict)
    
    Returns:
        Dictionary containing training results and metrics
    """
    
    try:
        # Initialize YOLO11n-seg model
        model = YOLO("yolo11n-seg.pt")
        
        # Prepare training arguments - only include parameters that are explicitly set
        # YOLO will use its own defaults for parameters not specified
        train_args = {
            "data": config.get("data_yml"),
            #"project": "runs/train",
            #"name": "seg",
            "exist_ok": False,
            "save": True,
            "save_period": 10,  # Save checkpoint every 10 epochs
        }
        
        # Only add parameters if they are explicitly set in config
        if "epochs" in config:
            train_args["epochs"] = config.get("epochs")
        if "image_size" in config:
            train_args["imgsz"] = config.get("image_size")
        if "batch_size" in config:
            train_args["batch"] = config.get("batch_size")
        if "learning_rate" in config:
            train_args["lr0"] = config.get("learning_rate")
        if "patience" in config:
            train_args["patience"] = config.get("patience")
        if "weight_decay" in config:
            train_args["weight_decay"] = config.get("weight_decay")
        if "momentum" in config:
            train_args["momentum"] = config.get("momentum")
        
        # Set optimizer only if specified
        if "optimizer" in config:
            optimizer = config.get("optimizer", "Adam").lower()
            if optimizer == "sgd":
                train_args["optimizer"] = "SGD"
            elif optimizer == "adam":
                train_args["optimizer"] = "Adam"
            elif optimizer == "adamw":
                train_args["optimizer"] = "AdamW"
            elif optimizer == "rmsprop":
                train_args["optimizer"] = "RMSprop"
        
        # Train the model
        results = model.train(**train_args)
        
        # Get the path to the latest training run
        from utils.yolo_metrics import find_latest_yolo_run, load_yolo_results, extract_yolo_metrics
        
        # Look in runs/segment for training runs
        latest_run = find_latest_yolo_run("runs/segment")
        if latest_run is None:
            # Try alternative locations
            latest_run = find_latest_yolo_run("runs/train")
            if latest_run is None:
                latest_run = find_latest_yolo_run()
        
        # Load results and extract metrics
        results_df = load_yolo_results(latest_run) if latest_run else None
        metrics = extract_yolo_metrics(results_df) if results_df is not None else {}
        
        # Get best weights path
        if latest_run:
            best_weights = latest_run / "weights" / "best.pt"
            if not best_weights.exists():
                best_weights = latest_run / "best.pt"
        else:
            best_weights = Path("runs/segment/train/weights/best.pt")
        
        return {
            "status": "completed",
            "best_weights": str(best_weights),
            "final_epoch": config.get("epochs", 100),
            "metrics": metrics,
            "results_path": str(latest_run) if latest_run else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "best_weights": None
        }

