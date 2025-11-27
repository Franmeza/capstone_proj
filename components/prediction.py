"""
Prediction component for running inference on images, videos, and live camera.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import time
from utils.model import predict_image_or_video, predict_frame, get_prediction_model, clear_model_cache
from utils.yolo_metrics import find_latest_yolo_run

def render_prediction():
    """Render the prediction page for image/video/camera inference."""
    
    st.title("🔮 Prediction")
    st.markdown("Detect corrosion in images, videos, or live camera feed using YOLO segmentation")
    
    # Initialize session state for camera
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "last_prediction_result" not in st.session_state:
        st.session_state.last_prediction_result = None
    
    # Model selection
    st.sidebar.subheader("🎯 Model Settings")
    
    # Find available weights
    weights_options = {"Pre-trained (yolo11n-seg.pt)": None}
    
    latest_run = find_latest_yolo_run("runs/segment")
    if latest_run:
        best_weights = latest_run / "weights" / "best.pt"
        if best_weights.exists():
            weights_options[f"Latest Training ({latest_run.name})"] = str(best_weights)
    
    # Allow custom weights path
    custom_weights = st.sidebar.text_input(
        "Custom Weights Path (optional)",
        placeholder="path/to/weights.pt",
        help="Enter path to custom YOLO weights file"
    )
    if custom_weights and Path(custom_weights).exists():
        weights_options["Custom Weights"] = custom_weights
    
    selected_weights_label = st.sidebar.selectbox(
        "Select Model Weights",
        options=list(weights_options.keys()),
        help="Choose which model weights to use for prediction"
    )
    selected_weights = weights_options[selected_weights_label]
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    if st.sidebar.button("🔄 Clear Model Cache"):
        clear_model_cache()
        st.sidebar.success("Model cache cleared!")
    
    st.markdown("---")
    
    # Input method selection
    input_method = st.radio(
        "Input Source",
        ["📷 Image Upload", "🎬 Video Upload", "📁 File Path", "📹 Live Camera"],
        horizontal=True
    )
    
    if input_method == "📷 Image Upload":
        render_image_upload(selected_weights, conf_threshold)
    elif input_method == "🎬 Video Upload":
        render_video_upload(selected_weights, conf_threshold)
    elif input_method == "📁 File Path":
        render_file_path(selected_weights, conf_threshold)
    elif input_method == "📹 Live Camera":
        render_live_camera(selected_weights, conf_threshold)


def render_image_upload(weights_path: str, conf_threshold: float):
    """Handle image file upload and prediction."""
    
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload an image file to analyze for corrosion"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Original Image")
            original_img = Image.open(uploaded_file)
            st.image(original_img, caption="Uploaded Image", width='stretch')
        
        # Run prediction button
        if st.button("🚀 Run Prediction", type="primary", width='stretch'):
            with st.spinner("Running YOLO segmentation..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = Path("width='stretch'd")
                    temp_path.mkdir(exist_ok=True)
                    file_ext = uploaded_file.name.split(".")[-1]
                    temp_file = temp_path / f"upload.{file_ext}"
                    
                    # Reset file position and save
                    uploaded_file.seek(0)
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Run prediction
                    result = predict_image_or_video(
                        str(temp_file), 
                        file_type="image",
                        weights_path=weights_path,
                        conf_threshold=conf_threshold
                    )
                    
                    st.session_state.last_prediction_result = result
                    
                    # Cleanup temp file
                    if temp_file.exists():
                        temp_file.unlink()
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
                    return
        
        # Display results if available
        if st.session_state.last_prediction_result:
            result = st.session_state.last_prediction_result
            
            with col2:
                st.subheader("🎨 Prediction Result")
                if result.get('output_path') and Path(result['output_path']).exists():
                    result_img = Image.open(result['output_path'])
                    st.image(result_img, caption="Segmentation Result", width='stretch')
            
            st.markdown("---")
            display_prediction_stats(result)
    else:
        st.info("👆 Upload an image file above to get started")
        show_recent_predictions()


def render_video_upload(weights_path: str, conf_threshold: float):
    """Handle video file upload and prediction."""
    
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Upload a video file to analyze for corrosion"
    )
    
    if uploaded_file is not None:
        st.subheader("📥 Uploaded Video")
        st.video(uploaded_file)
        
        if st.button("🚀 Process Video", type="primary", width='stretch'):
            with st.spinner("Processing video with YOLO segmentation... This may take a while."):
                try:
                    # Save uploaded file temporarily
                    temp_path = Path("temp_upload")
                    temp_path.mkdir(exist_ok=True)
                    file_ext = uploaded_file.name.split(".")[-1]
                    temp_file = temp_path / f"upload_video.{file_ext}"
                    
                    uploaded_file.seek(0)
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Run prediction
                    result = predict_image_or_video(
                        str(temp_file),
                        file_type="video",
                        weights_path=weights_path,
                        conf_threshold=conf_threshold
                    )
                    
                    st.session_state.last_prediction_result = result
                    
                    # Cleanup temp file
                    if temp_file.exists():
                        temp_file.unlink()
                    
                except Exception as e:
                    st.error(f"Error during video processing: {str(e)}")
                    st.exception(e)
                    return
        
        # Display results if available
        if st.session_state.last_prediction_result:
            result = st.session_state.last_prediction_result
            
            st.markdown("---")
            st.subheader("🎨 Processed Video")
            
            if result.get('output_path') and Path(result['output_path']).exists():
                st.video(result['output_path'])
                
                # Download button
                with open(result['output_path'], "rb") as f:
                    st.download_button(
                        "📥 Download Processed Video",
                        f.read(),
                        file_name=f"processed_video.mp4",
                        mime="video/mp4"
                    )
            
            display_prediction_stats(result, is_video=True)
    else:
        st.info("👆 Upload a video file above to get started")


def render_file_path(weights_path: str, conf_threshold: float):
    """Handle file path input for prediction."""
    
    file_path = st.text_input(
        "File Path",
        placeholder="Enter path to image or video file",
        help="Enter the full path to your image or video file"
    )
    
    if file_path:
        path = Path(file_path)
        
        if not path.exists():
            st.error(f"File not found: {file_path}")
            return
        
        # Determine file type
        ext = path.suffix.lower()
        is_image = ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        is_video = ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        
        if not is_image and not is_video:
            st.error(f"Unsupported file format: {ext}")
            return
        
        file_type = "image" if is_image else "video"
        
        # Display preview
        st.subheader("📥 Preview")
        if is_image:
            img = Image.open(file_path)
            st.image(img, caption="Input Image", width='stretch')
        else:
            st.video(file_path)
        
        if st.button("🚀 Run Prediction", type="primary", width='stretch'):
            with st.spinner(f"Processing {file_type}..."):
                try:
                    result = predict_image_or_video(
                        file_path,
                        file_type=file_type,
                        weights_path=weights_path,
                        conf_threshold=conf_threshold
                    )
                    
                    st.session_state.last_prediction_result = result
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
                    return
        
        # Display results
        if st.session_state.last_prediction_result:
            result = st.session_state.last_prediction_result
            
            st.markdown("---")
            st.subheader("🎨 Result")
            
            if result.get('output_path') and Path(result['output_path']).exists():
                if is_image:
                    result_img = Image.open(result['output_path'])
                    st.image(result_img, caption="Prediction Result", width='stretch')
                else:
                    st.video(result['output_path'])
            
            display_prediction_stats(result, is_video=is_video)
    else:
        st.info("👆 Enter a file path above to get started")


def render_live_camera(weights_path: str, conf_threshold: float):
    """Handle live camera prediction with real-time display."""
    
    st.markdown("""
    ### 📹 Live Camera Detection
    
    Start the camera to detect corrosion in real-time using YOLO segmentation.
    """)
    
    # Camera settings
    col1, col2 = st.columns(2)
    with col1:
        camera_index = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=10,
            value=0,
            help="Camera device index (0 for default camera)"
        )
    with col2:
        frame_skip = st.number_input(
            "Process Every N Frames",
            min_value=1,
            max_value=10,
            value=1,
            help="Process every Nth frame for better performance"
        )
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        start_btn = st.button("▶️ Start Camera", type="primary", width='stretch')
    with col2:
        stop_btn = st.button("⏹️ Stop Camera", type="secondary", width='stretch')
    
    if start_btn:
        st.session_state.camera_running = True
    if stop_btn:
        st.session_state.camera_running = False
    
    # Placeholders for live display
    st.markdown("---")
    
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if st.session_state.camera_running:
        # Load model once
        try:
            model = get_prediction_model(weights_path)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.session_state.camera_running = False
            return
        
        # Open camera
        cap = cv2.VideoCapture(int(camera_index))
        
        if not cap.isOpened():
            st.error(f"Could not open camera {camera_index}. Please check if the camera is connected.")
            st.session_state.camera_running = False
            return
        
        frame_count = 0
        last_stats = None
        last_annotated = None
        
        try:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Process frame
                if frame_count % frame_skip == 0:
                    annotated_frame, stats = predict_frame(frame, model, conf_threshold)
                    last_annotated = annotated_frame
                    last_stats = stats
                else:
                    # Use last prediction for skipped frames
                    if last_annotated is not None:
                        annotated_frame = last_annotated
                        stats = last_stats
                    else:
                        annotated_frame = frame
                        stats = {"corroded_area_pct": 0, "confidence": 0, "num_regions": 0, "severity": "None"}
                
                # Convert BGR to RGB for display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(display_frame, caption="Live Camera Feed", width='stretch')
                
                # Display stats
                with stats_placeholder.container():
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Corrosion Area", f"{stats['corroded_area_pct']:.1f}%")
                    with cols[1]:
                        st.metric("Confidence", f"{stats['confidence']:.2f}")
                    with cols[2]:
                        st.metric("Regions", stats['num_regions'])
                    with cols[3]:
                        severity_color = {
                            "None": "🟢",
                            "Low": "🟡", 
                            "Medium": "🟠",
                            "High": "🔴"
                        }
                        st.metric("Severity", f"{severity_color.get(stats['severity'], '⚪')} {stats['severity']}")
                
                # Small delay to prevent overwhelming the display
                time.sleep(0.03)
                
                # Check if stop was pressed (need to rerun to check)
                # This is a workaround since Streamlit doesn't have true async
                
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
        finally:
            cap.release()
            st.session_state.camera_running = False
    else:
        # Show placeholder when camera is not running
        frame_placeholder.info("📷 Camera is not active. Click 'Start Camera' to begin live detection.")
        
        with stats_placeholder.container():
            cols = st.columns(4)
            with cols[0]:
                st.metric("Corrosion Area", "- %")
            with cols[1]:
                st.metric("Confidence", "-")
            with cols[2]:
                st.metric("Regions", "-")
            with cols[3]:
                st.metric("Severity", "⚪ -")


def display_prediction_stats(result: dict, is_video: bool = False):
    """Display prediction statistics in a nice format."""
    
    st.subheader("📊 Detection Statistics")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(
            "Corrosion Area",
            f"{result.get('corroded_area_pct', 0):.2f}%",
            help="Percentage of the image/frame covered by detected corrosion"
        )
    
    with cols[1]:
        st.metric(
            "Confidence",
            f"{result.get('confidence', 0):.2f}",
            help="Average confidence score of detections"
        )
    
    with cols[2]:
        st.metric(
            "Detected Regions",
            result.get('num_regions', 0),
            help="Number of corrosion regions detected"
        )
    
    with cols[3]:
        severity = result.get('severity', 'Unknown')
        severity_colors = {
            "None": "🟢",
            "Low": "🟡",
            "Medium": "🟠",
            "High": "🔴"
        }
        st.metric(
            "Severity",
            f"{severity_colors.get(severity, '⚪')} {severity}",
            help="Corrosion severity based on area percentage"
        )
    
    # Show detailed detections if available
    if result.get('detections'):
        with st.expander("📋 Detailed Detections"):
            for i, det in enumerate(result['detections']):
                st.markdown(f"""
                **Detection {i+1}:**
                - Class: `{det.get('class', 'Unknown')}`
                - Confidence: `{det.get('confidence', 0):.3f}`
                - Area: `{det.get('area_pct', 0):.2f}%` ({det.get('area_pixels', 0):,} pixels)
                """)
    
    # Video-specific info
    if is_video and result.get('total_frames'):
        st.info(f"📹 Processed {result['total_frames']} frames at {result.get('fps', 30)} FPS")
    
    # Download processed result
    if result.get('output_path') and Path(result['output_path']).exists():
        output_path = Path(result['output_path'])
        
        with open(output_path, "rb") as f:
            mime_type = "image/png" if not is_video else "video/mp4"
            st.download_button(
                "📥 Download Processed Result",
                f.read(),
                file_name=output_path.name,
                mime=mime_type
            )


def show_recent_predictions():
    """Show recent prediction results."""
    
    predict_dir = Path("runs/predict")
    
    # Show recent predictions
    if predict_dir.exists():
        recent_preds = sorted(
            [f for f in predict_dir.glob("processed_*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:6]
        
        if recent_preds:
            st.markdown("### 🕐 Recent Predictions")
            
            cols = st.columns(min(len(recent_preds), 3))
            for i, pred_file in enumerate(recent_preds[:6]):
                col_idx = i % 3
                
                # Start new row after 3 items
                if i == 3:
                    cols = st.columns(min(len(recent_preds) - 3, 3))
                
                with cols[col_idx]:
                    try:
                        img = Image.open(pred_file)
                        st.image(img, caption=pred_file.name, width='stretch')
                        
                        # Show file modification time
                        from datetime import datetime
                        mod_time = pred_file.stat().st_mtime
                        mod_datetime = datetime.fromtimestamp(mod_time)
                        st.caption(f"📅 {mod_datetime.strftime('%Y-%m-%d %H:%M')}")
                    except Exception as e:
                        st.error(f"Error loading {pred_file.name}")
