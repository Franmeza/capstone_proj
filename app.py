"""
Main Streamlit application entry point for YOLO Corrosion Detection UI.
"""

import streamlit as st

# Import page components
from components.dashboard import render_dashboard
from components.prediction import render_prediction
from components.training import render_training

# Page configuration
st.set_page_config(
    page_title="YOLO Corrosion Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, minimal design
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #4CAF50;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    h2 {
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("🔍 Corrosion Detection")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Prediction", "Training"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "YOLO-based segmentation system for detecting corrosion in metals. "
        "Upload images or videos to detect and analyze corrosion patterns."
    )
    
    # Route to appropriate page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Prediction":
        render_prediction()
    elif page == "Training":
        render_training()

if __name__ == "__main__":
    main()

