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

# Global model cache to avoid reloading
_prediction_model = None

def get_prediction_model(weights_path: str = None) -> YOLO:
    """
    Get or load YOLO model for prediction.
    Uses cached model if already loaded.
    
    Args:
        weights_path: Path to custom weights file. If None, uses best.pt from latest training
                     or falls back to yolo11n-seg.pt
    
    Returns:
        YOLO model instance
    """
    global _prediction_model
    
    if weights_path:
        # Load specific weights
        return YOLO(weights_path)
    
    if _prediction_model is not None:
        return _prediction_model
    
    # Try to find best.pt from latest training
    from utils.yolo_metrics import find_latest_yolo_run
    
    latest_run = find_latest_yolo_run("runs/segment")
    if latest_run:
        best_weights = latest_run / "weights" / "best.pt"
        if best_weights.exists():
            _prediction_model = YOLO(str(best_weights))
            return _prediction_model
    
    # Fall back to pretrained model
    _prediction_model = YOLO("yolo11n-seg.pt")
    return _prediction_model


def predict_image_or_video(input_path: str, file_type: str = "image", 
                           weights_path: str = None, conf_threshold: float = 0.25) -> dict:
    """
    Run YOLO segmentation inference on images or videos.
    
    Args:
        input_path: Path to input image or video file
        file_type: "image" or "video"
        weights_path: Optional path to custom weights file
        conf_threshold: Confidence threshold for predictions
    
    Returns:
        Dictionary containing:
        - output_path: Path to processed output file
        - corroded_area_pct: Percentage of image/video with corrosion
        - confidence: Average confidence score
        - num_regions: Number of detected corrosion regions
        - severity: Severity level (Low/Medium/High)
        - detections: List of detection details
    """
    
    output_dir = Path("runs/predict")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = get_prediction_model(weights_path)
    
    if file_type == "image":
        return _predict_image(model, input_path, output_dir, conf_threshold)
    else:
        return _predict_video(model, input_path, output_dir, conf_threshold)


def _predict_image(model: YOLO, input_path: str, output_dir: Path, 
                   conf_threshold: float = 0.25) -> dict:
    """
    Run YOLO inference on a single image.
    """
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    h, w = img.shape[:2]
    total_pixels = h * w
    
    # Run inference
    results = model.predict(source=input_path, conf=conf_threshold, save=False, verbose=False)
    
    # Process results
    result = results[0]
    annotated_img = result.plot()  # Get annotated image with masks
    
    # Calculate statistics
    corroded_pixels = 0
    confidences = []
    num_regions = 0
    detections = []
    
    if result.masks is not None and len(result.masks) > 0:
        num_regions = len(result.masks)
        
        for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
            # Get mask area
            mask_np = mask.cpu().numpy()
            mask_area = np.sum(mask_np > 0.5)
            corroded_pixels += mask_area
            
            # Get confidence
            conf = float(box.conf[0])
            confidences.append(conf)
            
            # Get class name
            class_id = int(box.cls[0])
            class_name = model.names[class_id] if class_id < len(model.names) else f"Class_{class_id}"
            
            detections.append({
                "class": class_name,
                "confidence": conf,
                "area_pixels": int(mask_area),
                "area_pct": (mask_area / total_pixels) * 100
            })
    
    # Calculate overall statistics
    corroded_area_pct = (corroded_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0.0
    severity = "None" if num_regions == 0 else "Low" if corroded_area_pct < 20 else "Medium" if corroded_area_pct < 35 else "High"
    
    # Save result
    output_path = output_dir / f"processed_{Path(input_path).name}"
    cv2.imwrite(str(output_path), annotated_img)
    
    return {
        "output_path": str(output_path),
        "corroded_area_pct": corroded_area_pct,
        "confidence": avg_confidence,
        "num_regions": num_regions,
        "severity": severity,
        "detections": detections,
        "image_size": (w, h)
    }


def _predict_video(model: YOLO, input_path: str, output_dir: Path,
                   conf_threshold: float = 0.25) -> dict:
    """
    Run YOLO inference on a video file.
    """
    import subprocess
    import shutil
    
    # Load video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not load video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_pixels = width * height
    
    # Create temporary output path for OpenCV (will be re-encoded)
    temp_output_path = output_dir / f"temp_processed_{Path(input_path).stem}.avi"
    final_output_path = output_dir / f"processed_{Path(input_path).stem}.mp4"
    
    # Use XVID codec for temp file (widely supported by OpenCV)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    total_corroded_pct = 0
    all_confidences = []
    total_regions = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference on frame
        results = model.predict(source=frame, conf=conf_threshold, save=False, verbose=False)
        result = results[0]
        
        # Get annotated frame
        annotated_frame = result.plot()
        out.write(annotated_frame)
        
        # Calculate frame statistics
        if result.masks is not None and len(result.masks) > 0:
            for mask, box in zip(result.masks.data, result.boxes):
                mask_np = mask.cpu().numpy()
                mask_area = np.sum(mask_np > 0.5)
                total_corroded_pct += (mask_area / total_pixels) * 100
                all_confidences.append(float(box.conf[0]))
                total_regions += 1
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Re-encode video to H.264 for web browser compatibility
    output_path = _convert_video_to_h264(temp_output_path, final_output_path)
    
    # Calculate average statistics
    corroded_area_pct = total_corroded_pct / frame_count if frame_count > 0 else 0
    avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
    avg_regions = total_regions / frame_count if frame_count > 0 else 0
    severity = "None" if avg_regions == 0 else "Low" if corroded_area_pct < 20 else "Medium" if corroded_area_pct < 35 else "High"
    
    return {
        "output_path": str(output_path),
        "corroded_area_pct": corroded_area_pct,
        "confidence": avg_confidence,
        "num_regions": int(avg_regions),
        "severity": severity,
        "total_frames": frame_count,
        "video_size": (width, height),
        "fps": fps
    }


def _convert_video_to_h264(input_path: Path, output_path: Path) -> Path:
    """
    Convert video to H.264 codec for web browser compatibility.
    Uses ffmpeg (via imageio-ffmpeg) for conversion.
    """
    import subprocess
    import shutil
    
    # Try to get ffmpeg path from imageio-ffmpeg (bundled ffmpeg)
    ffmpeg_path = None
    
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        # imageio-ffmpeg not installed, try system ffmpeg
        ffmpeg_path = shutil.which("ffmpeg")
    except Exception:
        ffmpeg_path = shutil.which("ffmpeg")
    
    if ffmpeg_path:
        try:
            # Use ffmpeg to convert to H.264 with web-compatible settings
            cmd = [
                ffmpeg_path,
                "-y",  # Overwrite output file
                "-i", str(input_path),
                "-c:v", "libx264",  # H.264 codec
                "-preset", "fast",  # Encoding speed
                "-crf", "23",  # Quality (lower = better, 18-28 is good range)
                "-pix_fmt", "yuv420p",  # Pixel format for compatibility
                "-movflags", "+faststart",  # Enable streaming
                "-an",  # No audio (since we're processing video only)
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Remove temp file
            if input_path.exists():
                input_path.unlink()
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg conversion failed: {e}")
            # Fall through to fallback
        except Exception as e:
            print(f"Error during video conversion: {e}")
            # Fall through to fallback
    
    # Fallback: rename temp file to output (won't be web-compatible but at least exists)
    if input_path.exists():
        # Change extension to .avi since it wasn't converted
        fallback_path = output_path.with_suffix('.avi')
        shutil.move(str(input_path), str(fallback_path))
        return fallback_path
    
    return input_path


def predict_frame(frame: np.ndarray, model: YOLO = None, 
                  conf_threshold: float = 0.25) -> tuple:
    """
    Run YOLO inference on a single frame (for live camera).
    
    Args:
        frame: Input frame as numpy array (BGR)
        model: YOLO model instance. If None, uses cached model
        conf_threshold: Confidence threshold
    
    Returns:
        Tuple of (annotated_frame, stats_dict)
    """
    if model is None:
        model = get_prediction_model()
    
    h, w = frame.shape[:2]
    total_pixels = h * w
    
    # Run inference
    results = model.predict(source=frame, conf=conf_threshold, save=False, verbose=False)
    result = results[0]
    
    # Get annotated frame
    annotated_frame = result.plot()
    
    # Calculate statistics
    corroded_pixels = 0
    confidences = []
    num_regions = 0
    
    if result.masks is not None and len(result.masks) > 0:
        num_regions = len(result.masks)
        
        for mask, box in zip(result.masks.data, result.boxes):
            mask_np = mask.cpu().numpy()
            corroded_pixels += np.sum(mask_np > 0.5)
            confidences.append(float(box.conf[0]))
    
    corroded_area_pct = (corroded_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0.0
    severity = "None" if num_regions == 0 else "Low" if corroded_area_pct < 20 else "Medium" if corroded_area_pct < 35 else "High"
    
    stats = {
        "corroded_area_pct": corroded_area_pct,
        "confidence": avg_confidence,
        "num_regions": num_regions,
        "severity": severity
    }
    
    return annotated_frame, stats


def clear_model_cache():
    """Clear the cached prediction model."""
    global _prediction_model
    _prediction_model = None


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

