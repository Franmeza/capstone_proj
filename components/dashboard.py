"""
Dashboard component displaying metrics, charts, and sample predictions.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import numpy as np
from utils.yolo_metrics import get_yolo_metrics, get_training_data_from_yolo, find_latest_yolo_run

def render_dashboard():
    """Render the dashboard page with metrics and visualizations."""
    
    st.title("📊 Dashboard")
    st.markdown("Overview of model performance and sample predictions")
    
    # Get real YOLO metrics
    yolo_metrics, yolo_df = get_yolo_metrics()
    
    # Use real metrics if available, otherwise use defaults
    if yolo_metrics:
        metrics = {
            "accuracy": yolo_metrics.get("accuracy", 0.0),
            "precision": yolo_metrics.get("precision", 0.0),
            "recall": yolo_metrics.get("recall", 0.0),
            "mAP50": yolo_metrics.get("map50", 0.0),
            "mAP50-95": yolo_metrics.get("map50_95", 0.0)
        }
        
        # Show info about which run we're displaying
        latest_run = find_latest_yolo_run("runs/segment")
        if latest_run is None:
            latest_run = find_latest_yolo_run()
        if latest_run:
            with st.expander("ℹ️ Training Run Info"):
                st.text(f"Path: {latest_run}")
                if yolo_df is not None:
                    st.text(f"Total Epochs: {len(yolo_df)}")
                    st.text(f"Available Columns: {', '.join(yolo_df.columns.tolist()[:10])}")
                    if len(yolo_df.columns) > 10:
                        st.text(f"... and {len(yolo_df.columns) - 10} more columns")
    else:
        # Fallback to default values if no training results found
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50-95": 0.0
        }
        st.warning("⚠️ No YOLO training results found. Train a model first to see real metrics. Looking in `runs/segment/train*/results.csv`")
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate deltas if we have previous metrics (for now, show None if no comparison)
    delta_acc = None if not yolo_metrics else None  # Could compare with previous run
    delta_prec = None if not yolo_metrics else None
    delta_rec = None if not yolo_metrics else None
    delta_iou = None if not yolo_metrics else None
    delta_map = None if not yolo_metrics else None
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%", delta=delta_acc)
    with col2:
        st.metric("Precision", f"{metrics['precision']*100:.1f}%", delta=delta_prec)
    with col3:
        st.metric("Recall", f"{metrics['recall']*100:.1f}%", delta=delta_rec)
    with col4:
        st.metric("mAP50", f"{metrics['mAP50']*100:.1f}%", delta=delta_iou)
    with col5:
        st.metric("mAP50-95", f"{metrics['mAP50-95']*100:.1f}%", delta=delta_map)
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Training Progress")
        
        # Get real training data from YOLO
        if yolo_df is not None:
            training_data = get_training_data_from_yolo(yolo_df)
            epochs = training_data.get("epoch", list(range(len(yolo_df))))
            train_loss = training_data.get("train_loss", [])
            val_loss = training_data.get("val_loss", [])
            train_acc = training_data.get("train_accuracy", training_data.get("accuracy", []))
            val_acc = training_data.get("val_accuracy", training_data.get("accuracy", []))
        else:
            # Fallback to dummy data
            epochs = list(range(1, 21))
            train_loss = []
            val_loss = []
            train_acc = []
            val_acc = []
            st.caption("No training data available. Train a model to see progress charts.")
        
        # Loss chart
        if train_loss or val_loss:
            fig_loss = go.Figure()
            if train_loss:
                fig_loss.add_trace(go.Scatter(
                    x=epochs, y=train_loss, name="Train Loss",
                    line=dict(color="#4CAF50", width=2)
                ))
            if val_loss:
                fig_loss.add_trace(go.Scatter(
                    x=epochs, y=val_loss, name="Validation Loss",
                    line=dict(color="#FF9800", width=2)
                ))
            fig_loss.update_layout(
                title="Loss per Epoch",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                hovermode="x unified",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig_loss, width='stretch')
        else:
            st.info("No loss data available")
        
        # Accuracy chart
        if train_acc or val_acc:
            fig_acc = go.Figure()
            if train_acc:
                fig_acc.add_trace(go.Scatter(
                    x=epochs, y=train_acc, name="Train Accuracy",
                    line=dict(color="#2196F3", width=2)
                ))
            if val_acc:
                fig_acc.add_trace(go.Scatter(
                    x=epochs, y=val_acc, name="Validation Accuracy",
                    line=dict(color="#9C27B0", width=2)
                ))
            fig_acc.update_layout(
                title="Accuracy per Epoch",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                hovermode="x unified",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig_acc, width='stretch')
        else:
            st.info("No accuracy data available")
    
    with col2:
        st.subheader("📊 Class Distribution")
        
        # Dummy class distribution
        class_data = pd.DataFrame({
            "Class": ["Rust", "Clean Metal", "Partial Corrosion"],
            "Count": [450, 320, 180],
            "Percentage": [47.4, 33.7, 18.9]
        })
        
        fig_bar = px.bar(
            class_data,
            x="Class",
            y="Count",
            color="Class",
            color_discrete_map={
                "Rust": "#d32f2f",
                "Clean Metal": "#4CAF50",
                "Partial Corrosion": "#FF9800"
            },
            text="Count"
        )
        fig_bar.update_layout(
            title="Sample Distribution by Class",
            xaxis_title="Class",
            yaxis_title="Count",
            template="plotly_white",
            height=300,
            showlegend=False
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, width='stretch')
        
        # mAP chart
        st.subheader("🎯 mAP Score")
        
        if yolo_df is not None:
            # Extract mAP data from YOLO results
            map_cols = [col for col in yolo_df.columns if "map" in col.lower()]
            if map_cols:
                # Use same epoch handling as training data
                if "epoch" in yolo_df.columns:
                    map_epochs = (yolo_df["epoch"] + 1).tolist()
                else:
                    map_epochs = [i + 1 for i in range(len(yolo_df))]
                map_values = yolo_df[map_cols[0]].tolist()
                
                fig_map = go.Figure()
                fig_map.add_trace(go.Scatter(
                    x=map_epochs,
                    y=map_values,
                    fill="tozeroy",
                    fillcolor="rgba(76, 175, 80, 0.2)",
                    line=dict(color="#4CAF50", width=2),
                    name="mAP"
                ))
                fig_map.update_layout(
                    title=f"Mean Average Precision ({map_cols[0]})",
                    xaxis_title="Epoch",
                    yaxis_title="mAP",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig_map, width='stretch')
            else:
                st.info("No mAP data found in results")
        else:
            st.info("No training data available for mAP chart")
    
    st.markdown("---")
    
    # Sample predictions preview
    st.subheader("🖼️ Sample Predictions")
    
    samples_dir = Path("data/samples")
    if samples_dir.exists():
        sample_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
        
        if sample_files:
            cols = st.columns(min(3, len(sample_files)))
            
            for idx, sample_file in enumerate(sample_files[:3]):
                with cols[idx % 3]:
                    try:
                        img = Image.open(sample_file)
                        st.image(img, caption=sample_file.name, width='stretch')
                        
                        # Dummy corrosion statistics
                        st.caption(f"Corroded Area: {np.random.randint(15, 45)}%")
                        st.caption(f"Confidence: {np.random.uniform(0.85, 0.98):.2f}")
                    except Exception as e:
                        st.error(f"Error loading {sample_file.name}: {str(e)}")
        else:
            st.info("No sample images found in data/samples/")
    else:
        st.info("Sample directory not found. Add images to data/samples/ for preview.")

