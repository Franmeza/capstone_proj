"""
Prediction component for running inference on images and videos.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
from utils.model import predict_image_or_video

def render_prediction():
    """Render the prediction page for image/video inference."""
    
    st.title("🔮 Prediction")
    st.markdown("Upload an image or video to detect corrosion")
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["File Upload", "Path Input"],
        horizontal=True
    )
    
    uploaded_file = None
    file_path = None
    
    if input_method == "File Upload":
        file_type = st.selectbox("File Type", ["Image", "Video"])
        
        if file_type == "Image":
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload an image file to analyze"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mov", "mkv"],
                help="Upload a video file to analyze"
            )
    else:
        file_path = st.text_input(
            "File Path",
            placeholder="Enter path to image or video file",
            help="Enter the full path to your image or video file"
        )
    
    st.markdown("---")
    
    # Run prediction button
    if st.button("🚀 Run Prediction", type="primary", width='stretch'):
        if uploaded_file is not None or (file_path and Path(file_path).exists()):
            with st.spinner("Running prediction..."):
                try:
                    # Prepare input
                    if uploaded_file:
                        # Save uploaded file temporarily
                        temp_path = Path("temp_upload")
                        temp_path.mkdir(exist_ok=True)
                        file_ext = uploaded_file.name.split(".")[-1]
                        temp_file = temp_path / f"upload.{file_ext}"
                        
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        input_path = str(temp_file)
                        file_type = "image" if file_ext.lower() in ["jpg", "jpeg", "png", "bmp"] else "video"
                    else:
                        input_path = file_path
                        file_ext = Path(file_path).suffix.lower()
                        file_type = "image" if file_ext in [".jpg", ".jpeg", ".png", ".bmp"] else "video"
                    
                    # Run prediction
                    result = predict_image_or_video(input_path, file_type)
                    
                    # Display results
                    st.success("Prediction completed!")
                    st.markdown("---")
                    
                    # Results section
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Statistics")
                        
                        if result:
                            st.metric("Corroded Area", f"{result.get('corroded_area_pct', 0):.2f}%")
                            st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                            st.metric("Detected Regions", result.get('num_regions', 0))
                            
                            if result.get('severity'):
                                st.metric("Severity", result['severity'])
                    
                    with col2:
                        st.subheader("📥 Download Results")
                        
                        if result and result.get('output_path'):
                            output_path = result['output_path']
                            
                            if file_type == "image":
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        "Download Processed Image",
                                        f.read(),
                                        file_name=f"processed_{Path(input_path).name}",
                                        mime="image/png"
                                    )
                            else:
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        "Download Processed Video",
                                        f.read(),
                                        file_name=f"processed_{Path(input_path).name}",
                                        mime="video/mp4"
                                    )
                    
                    # Display result
                    st.markdown("---")
                    st.subheader("🎨 Result with Segmentation Overlay")
                    
                    if file_type == "image" and result.get('output_path'):
                        result_img = Image.open(result['output_path'])
                        st.image(result_img, caption="Prediction Result", width='stretch')
                    elif file_type == "video" and result.get('output_path'):
                        st.video(result['output_path'])
                    
                    # Cleanup temp file
                    if uploaded_file and Path(input_path).exists():
                        Path(input_path).unlink()
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
        else:
            st.warning("Please upload a file or provide a valid file path")
    
    # Show example if no file uploaded
    if uploaded_file is None and (not file_path or not Path(file_path).exists()):
        st.info("👆 Upload an image or video file above to get started")
        
        # Show sample if available
        samples_dir = Path("data/samples")
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
            if sample_files:
                st.markdown("### Example Sample")
                with st.expander("View sample image"):
                    img = Image.open(sample_files[0])
                    st.image(img, caption="Sample Image", width='stretch')

