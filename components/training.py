"""
Training component for model training with hyperparameter configuration.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time
from utils.model import train_model
from utils.history import save_training_run, get_training_history, compare_runs
from utils.report import generate_training_report

def render_training():
    """Render the training page with configuration and progress tracking."""
    
    st.title("🏋️ Training")
    st.markdown("Configure and train your YOLO model")
    
    # Initialize session state for persistent training results
    if "last_training_data" not in st.session_state:
        st.session_state.last_training_data = None
    if "last_training_epoch" not in st.session_state:
        st.session_state.last_training_epoch = None
    if "last_training_expected" not in st.session_state:
        st.session_state.last_training_expected = None
    if "last_training_chart" not in st.session_state:
        st.session_state.last_training_chart = None
    if "download_weights_path" not in st.session_state:
        st.session_state.download_weights_path = None
    if "download_report_path" not in st.session_state:
        st.session_state.download_report_path = None
    
    # Training configuration
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data configuration (required)
        data_yml_path = st.text_input(
            "Data YAML Path *",
            value="",
            help="Path to your dataset configuration YAML file (required)"
        )
        
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=1000,
            value=100,
            step=10,
            help="Number of training epochs"
        )
        
        use_default_lr = st.checkbox("Use default learning rate", value=True, key="default_lr")
        learning_rate = None
        if not use_default_lr:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=1.0,
                value=0.01,
                step=0.001,
                format="%.2f",
                help="Initial learning rate (leave default if unchecked)"
            )
        else:
            st.caption("Learning Rate: Using YOLO default")
        
        use_default_batch = st.checkbox("Use default batch size", value=True, key="default_batch")
        batch_size = None
        if not use_default_batch:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=16,
                step=1,
                help="Batch size for training (leave default if unchecked)"
            )
        else:
            st.caption("Batch Size: Using YOLO default")
    
    with col2:
        use_default_optimizer = st.checkbox("Use default optimizer", value=True, key="default_optimizer")
        optimizer = None
        if not use_default_optimizer:
            optimizer = st.selectbox(
                "Optimizer",
                ["SGD", "Adam", "AdamW", "RMSprop"],
                index=1,
                help="Optimization algorithm (leave default if unchecked)"
            )
        else:
            st.caption("Optimizer: Using YOLO default")
        
        use_default_imgsz = st.checkbox("Use default image size", value=True, key="default_imgsz")
        image_size = None
        if not use_default_imgsz:
            image_size = st.number_input(
                "Image Size",
                min_value=320,
                max_value=1280,
                value=640,
                step=32,
                help="Input image size (must be multiple of 32, leave default if unchecked)"
            )
        else:
            st.caption("Image Size: Using YOLO default")
        
        # Advanced options
        with st.expander("Advanced Options"):
            use_default_patience = st.checkbox("Use default patience", value=True, key="default_patience")
            patience = None
            if not use_default_patience:
                patience = st.number_input(
                    "Early Stopping Patience",
                    min_value=0,
                    max_value=100,
                    value=50,
                    help="Epochs to wait before early stopping"
                )
            else:
                st.caption("Patience: Using YOLO default")
            
            use_default_wd = st.checkbox("Use default weight decay", value=True, key="default_wd")
            weight_decay = None
            if not use_default_wd:
                weight_decay = st.number_input(
                    "Weight Decay",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.0005,
                    step=0.0001,
                    format="%.4f"
                )
            else:
                st.caption("Weight Decay: Using YOLO default")
            
            use_default_momentum = st.checkbox("Use default momentum", value=True, key="default_momentum")
            momentum = None
            if not use_default_momentum:
                momentum = st.number_input(
                    "Momentum",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.937,
                    step=0.001,
                    format="%.3f"
                )
            else:
                st.caption("Momentum: Using YOLO default")
    
    st.markdown("---")
    
    # Training history section
    st.subheader("📚 Training History")
    
    history_tab1, history_tab2 = st.tabs(["Past Runs", "Compare Models"])
    
    with history_tab1:
        history_df = get_training_history()
        
        if history_df is not None and len(history_df) > 0:
            st.dataframe(
                history_df,
                width='stretch',
                hide_index=True
            )
            
            # Download history
            csv = history_df.to_csv(index=False)
            st.download_button(
                "Download History CSV",
                csv,
                file_name="training_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No training history available yet. Start a training run to see history.")
    
    with history_tab2:
        history_df = get_training_history()
        
        if history_df is not None and len(history_df) > 0:
            run_ids = history_df['run_id'].tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                run1 = st.selectbox("Select Run 1", run_ids, key="run1")
            with col2:
                run2 = st.selectbox("Select Run 2", run_ids, key="run2")
            
            if run1 and run2 and run1 != run2:
                comparison = compare_runs(run1, run2)
                
                if comparison:
                    st.subheader("📊 Comparison")
                    
                    # Metrics comparison
                    metrics_cols = st.columns(len(comparison.get('metrics', {})))
                    for idx, (metric, value) in enumerate(comparison.get('metrics', {}).items()):
                        with metrics_cols[idx]:
                            st.metric(metric, f"{value:.4f}")
                    
                    # Charts
                    if comparison.get('charts'):
                        for chart_name, fig in comparison['charts'].items():
                            st.plotly_chart(fig, width='stretch', key=f"comparison_chart_{chart_name}")
        else:
            st.info("Need at least 2 training runs to compare models.")
    
    st.markdown("---")
    
    # Display last training results if available (persistent across reruns)
    if st.session_state.last_training_data or st.session_state.last_training_chart:
        st.subheader("📊 Last Training Results")
        
        # Show epoch counter
        if st.session_state.last_training_epoch and st.session_state.last_training_expected:
            st.text(f"✅ Training Complete! Final Epoch: {st.session_state.last_training_epoch}/{st.session_state.last_training_expected}")
        
        # Show chart - use stored chart if available, otherwise recreate from data
        if st.session_state.last_training_chart:
            st.plotly_chart(st.session_state.last_training_chart, width='stretch', key="last_training_chart")
        elif st.session_state.last_training_data and st.session_state.last_training_data.get("epoch") and st.session_state.last_training_data.get("loss"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.last_training_data["epoch"],
                y=st.session_state.last_training_data["loss"],
                name="Loss",
                line=dict(color="#d32f2f", width=2)
            ))
            if st.session_state.last_training_data.get("accuracy"):
                fig.add_trace(go.Scatter(
                    x=st.session_state.last_training_data["epoch"],
                    y=st.session_state.last_training_data["accuracy"],
                    name="Accuracy",
                    line=dict(color="#4CAF50", width=2),
                    yaxis="y2"
                ))
            layout_dict = {
                "title": "Training Progress (Final)",
                "xaxis_title": "Epoch",
                "yaxis_title": "Loss",
                "template": "plotly_white",
                "height": 300
            }
            if st.session_state.last_training_data.get("accuracy"):
                layout_dict["yaxis2"] = dict(title="Accuracy", overlaying="y", side="right")
            fig.update_layout(**layout_dict)
            st.plotly_chart(fig, width='stretch', key="last_training_chart_recreated")
        
        st.markdown("---")
    
    # Persistent download section (shows after training completes)
    if st.session_state.download_weights_path or st.session_state.download_report_path:
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.download_weights_path and Path(st.session_state.download_weights_path).exists():
                with open(st.session_state.download_weights_path, "rb") as f:
                    weights_data = f.read()
                    st.download_button(
                        "Download Weights (best.pt)",
                        weights_data,
                        file_name="best.pt",
                        mime="application/octet-stream",
                        key="download_weights_btn"
                    )
            else:
                st.info("Weights file not available")
        
        with col2:
            if st.session_state.download_report_path and Path(st.session_state.download_report_path).exists():
                with open(st.session_state.download_report_path, "rb") as f:
                    report_data = f.read()
                    file_ext = Path(st.session_state.download_report_path).suffix
                    mime_type = "application/pdf" if file_ext == ".pdf" else "text/html"
                    st.download_button(
                        f"Download Training Report ({file_ext[1:].upper()})",
                        report_data,
                        file_name=Path(st.session_state.download_report_path).name,
                        mime=mime_type,
                        key="download_report_btn"
                    )
            else:
                st.info("Training report not available")
        
        st.markdown("---")
    
    # Start training
    st.subheader("🚀 Start Training")
    
    if st.button("▶️ Start Training", type="primary", width='stretch'):
        # Clear previous training results when starting new training
        st.session_state.last_training_data = None
        st.session_state.last_training_epoch = None
        st.session_state.last_training_expected = None
        st.session_state.last_training_chart = None
        st.session_state.download_weights_path = None
        st.session_state.download_report_path = None
        # Validate required field
        if not data_yml_path or not data_yml_path.strip():
            st.error("⚠️ Data YAML Path is required!")
        elif not Path(data_yml_path).exists() and data_yml_path != "coco8-seg.yaml":
            st.error(f"Data YAML file not found: {data_yml_path}")
        else:
            # Training configuration - only include parameters that are explicitly set
            config = {
                "data_yml": data_yml_path
            }
            
            # Only add parameters if they are not None (i.e., user unchecked "use default")
            if epochs is not None:
                config["epochs"] = int(epochs)
            if learning_rate is not None:
                config["learning_rate"] = learning_rate
            if batch_size is not None:
                config["batch_size"] = int(batch_size)
            if optimizer is not None:
                config["optimizer"] = optimizer
            if image_size is not None:
                config["image_size"] = int(image_size)
            if patience is not None:
                config["patience"] = int(patience)
            if weight_decay is not None:
                config["weight_decay"] = weight_decay
            if momentum is not None:
                config["momentum"] = momentum
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()  # Will show single-line epoch counter
            chart_placeholder = st.empty()
            
            logs = []  # Keep for compatibility but won't use for epoch logging
            training_data = {"epoch": [], "loss": [], "accuracy": []}
            
            try:
                # Start actual YOLO training
                status_text.info("🚀 Starting YOLO11n-seg training...")
                log_container.text("Initializing YOLO11n-seg model...")
                
                # Import here to avoid issues if ultralytics is not installed
                from utils.yolo_metrics import load_yolo_results, get_training_data_from_yolo
                import threading
                
                # Run training in a way that allows progress monitoring
                training_result = None
                training_error = None
                
                def run_training():
                    nonlocal training_result, training_error
                    try:
                        training_result = train_model(config)
                    except Exception as e:
                        training_error = str(e)
                
                # Get the timestamp before starting training to identify the new run
                training_start_time = time.time()
                
                # Find baseline - get the latest run before training starts
                from utils.yolo_metrics import find_latest_yolo_run
                baseline_run = find_latest_yolo_run("runs/segment")
                baseline_mtime = baseline_run.stat().st_mtime if baseline_run and baseline_run.exists() else 0
                baseline_epochs = 0
                if baseline_run and (baseline_run / "results.csv").exists():
                    try:
                        baseline_df = load_yolo_results(baseline_run)
                        if baseline_df is not None:
                            baseline_epochs = len(baseline_df)
                    except:
                        pass
                
                # Start training in background
                train_thread = threading.Thread(target=run_training, daemon=True)
                train_thread.start()
                
                # Wait a bit for training to start and create the directory
                time.sleep(5)  # Increased wait time for YOLO to create directory
                
                # Monitor training progress by reading results.csv
                current_training_run = None
                last_epoch_seen = 0
                last_file_mtime = 0
                training_started = False  # Flag to know when we've found the new training
                # Get expected epochs (use config value or a large number if using default)
                expected_epochs = config.get("epochs", 100)  # Default to 100 for progress calculation
                
                while train_thread.is_alive():
                    # Find all training runs
                    base_path = Path("runs/segment")
                    if not base_path.exists():
                        time.sleep(2)
                        continue
                    
                    # Find the newest training run that was created after training started
                    train_dirs = [
                        d for d in base_path.iterdir() 
                        if d.is_dir() and d.name.startswith("train") and (d / "results.csv").exists()
                    ]
                    
                    # Find the run that was created most recently (after our baseline)
                    # Use a strict filter: must be created after training started (with buffer)
                    new_runs = [
                        d for d in train_dirs 
                        if d.stat().st_mtime > training_start_time - 2 and d != baseline_run  # Created after training AND not baseline
                    ]
                    
                    # If no new runs yet, wait (don't show anything from previous training)
                    if not new_runs:
                        if not training_started:
                            status_text.text("Waiting for training to start...")
                        time.sleep(2)
                        continue
                    
                    # Get the most recent new run
                    current_run = max(new_runs, key=lambda x: x.stat().st_mtime)
                    
                    # Only process if this is a new training run (not the baseline)
                    # Double check: ensure it's different from baseline
                    if current_run != baseline_run:
                        results_file = current_run / "results.csv"
                        
                        if results_file.exists():
                            # Check if file was modified (new epoch written)
                            file_mtime = results_file.stat().st_mtime
                            
                            # Mark that we've found the new training
                            if not training_started:
                                training_started = True
                                logs = []  # Clear any previous logs
                                last_epoch_seen = 0  # Reset epoch counter
                                log_container.text("Starting new training run...")
                            
                            # Update if this is a new run OR if the file was modified (new epoch)
                            if current_run != current_training_run or file_mtime > last_file_mtime:
                                current_training_run = current_run
                                last_file_mtime = file_mtime
                                
                                try:
                                    df = load_yolo_results(current_run)
                                    if df is not None and len(df) > 0:
                                        # Update progress based on epochs completed
                                        current_epoch = len(df)
                                        
                                        # Only process if this is a new training run (not baseline)
                                        # We already checked current_run != baseline_run above, so this should be safe
                                        # But add extra safety: only show epochs that are from the new training
                                        # Reset epoch counter when we first detect new training
                                        if not training_started:
                                            # First time seeing this run - reset everything
                                            last_epoch_seen = 0
                                        
                                        # Only show epochs that are actually from this new training
                                        # If it's the baseline run, skip entirely (shouldn't happen due to check above)
                                        if current_run == baseline_run:
                                            continue
                                        
                                        # Process the new training data
                                        progress = min(current_epoch / expected_epochs, 0.99)  # Don't reach 100% until done
                                        progress_bar.progress(progress)
                                        
                                        # Get training data for charts
                                        training_data_dict = get_training_data_from_yolo(df)
                                        
                                        if training_data_dict.get("epoch"):
                                            training_data = {
                                                "epoch": training_data_dict["epoch"],
                                                "loss": training_data_dict.get("val_loss", training_data_dict.get("train_loss", [])),
                                                "accuracy": training_data_dict.get("accuracy", [])
                                            }
                                            
                                            # Update status with latest metrics
                                            if training_data["loss"]:
                                                latest_loss = training_data["loss"][-1]
                                                latest_acc = training_data["accuracy"][-1] if training_data["accuracy"] else 0.0
                                                status_text.text(f"Epoch {current_epoch}/{expected_epochs} - Loss: {latest_loss:.4f}, Accuracy: {latest_acc:.4f}")
                                            
                                            # Update epoch counter (single line that updates)
                                            if current_epoch > last_epoch_seen:
                                                last_epoch_seen = current_epoch
                                                # Update the single line counter instead of appending logs
                                                log_container.text(f"Epoch {current_epoch}/{expected_epochs}")
                                            
                                            # Always update chart with latest data (every check, not just new epochs)
                                            if training_data["epoch"] and training_data["loss"]:
                                                fig = go.Figure()
                                                fig.add_trace(go.Scatter(
                                                    x=training_data["epoch"],
                                                    y=training_data["loss"],
                                                    name="Loss",
                                                    line=dict(color="#d32f2f", width=2)
                                                ))
                                                if training_data["accuracy"]:
                                                    fig.add_trace(go.Scatter(
                                                        x=training_data["epoch"],
                                                        y=training_data["accuracy"],
                                                        name="Accuracy",
                                                        line=dict(color="#4CAF50", width=2),
                                                        yaxis="y2"
                                                    ))
                                                layout_dict = {
                                                    "title": "Training Progress",
                                                    "xaxis_title": "Epoch",
                                                    "yaxis_title": "Loss",
                                                    "template": "plotly_white",
                                                    "height": 300
                                                }
                                                if training_data["accuracy"]:
                                                    layout_dict["yaxis2"] = dict(title="Accuracy", overlaying="y", side="right")
                                                fig.update_layout(**layout_dict)
                                                chart_placeholder.plotly_chart(fig, width='stretch', key="training_progress_chart")
                                except Exception as e:
                                    pass  # Continue monitoring even if reading fails
                    
                    time.sleep(2)  # Check every 2 seconds
                
                # Training complete
                progress_bar.progress(1.0)
                
                if training_error:
                    st.error(f"Training error: {training_error}")
                    raise Exception(training_error)
                
                if training_result is None:
                    st.error("Training completed but no results returned")
                    raise Exception("Training failed")
                
                if training_result.get("status") == "error":
                    st.error(f"Training error: {training_result.get('error', 'Unknown error')}")
                    raise Exception(training_result.get("error", "Training failed"))
                
                status_text.success("✅ Training completed!")
                
                # Wait a moment to ensure results.csv is fully written
                time.sleep(1)
                
                # Get final metrics from YOLO results and update final chart
                # Use the current_training_run we tracked during monitoring, or find the latest run
                final_run_path = None
                if current_training_run and current_training_run.exists() and (current_training_run / "results.csv").exists():
                    final_run_path = current_training_run
                elif training_result.get("results_path"):
                    results_path_obj = Path(training_result["results_path"])
                    if results_path_obj.exists() and (results_path_obj / "results.csv").exists():
                        final_run_path = results_path_obj
                
                # Fallback: find the latest run
                if final_run_path is None:
                    final_run_path = find_latest_yolo_run("runs/segment")
                    if final_run_path is None:
                        final_run_path = find_latest_yolo_run("runs/train")
                
                # Initialize final_df to None
                final_df = None
                
                if final_run_path and (final_run_path / "results.csv").exists():
                    final_df = load_yolo_results(final_run_path)
                    if final_df is not None and len(final_df) > 0:
                        # Verify we're reading from the correct file by checking the path
                        st.info(f"📊 Loading results from: {final_run_path} ({len(final_df)} epochs)")
                        
                        training_data_dict = get_training_data_from_yolo(final_df)
                        training_data = {
                            "epoch": training_data_dict.get("epoch", []),
                            "loss": training_data_dict.get("val_loss", training_data_dict.get("train_loss", [])),
                            "accuracy": training_data_dict.get("accuracy", [])
                        }
                        
                        # Update final epoch counter
                        final_epoch = len(final_df)
                        log_container.text(f"✅ Training Complete! Final Epoch: {final_epoch}/{expected_epochs}")
                        
                        # Store training data in session state for persistence
                        st.session_state.last_training_data = training_data
                        st.session_state.last_training_epoch = final_epoch
                        st.session_state.last_training_expected = expected_epochs
                        
                        # Display final chart in placeholder (keep it visible after training)
                        if training_data["epoch"] and training_data["loss"]:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=training_data["epoch"],
                                y=training_data["loss"],
                                name="Loss",
                                line=dict(color="#d32f2f", width=2)
                            ))
                            if training_data["accuracy"]:
                                fig.add_trace(go.Scatter(
                                    x=training_data["epoch"],
                                    y=training_data["accuracy"],
                                    name="Accuracy",
                                    line=dict(color="#4CAF50", width=2),
                                    yaxis="y2"
                                ))
                            layout_dict = {
                                "title": f"Training Progress (Final) - {final_run_path.name}",
                                "xaxis_title": "Epoch",
                                "yaxis_title": "Loss",
                                "template": "plotly_white",
                                "height": 300
                            }
                            if training_data["accuracy"]:
                                layout_dict["yaxis2"] = dict(title="Accuracy", overlaying="y", side="right")
                            fig.update_layout(**layout_dict)
                            chart_placeholder.plotly_chart(fig, width='stretch', key="training_final_chart")
                            
                            # Also store the figure in session state for persistence
                            st.session_state.last_training_chart = fig
                    else:
                        st.warning(f"⚠️ Could not load training results from: {final_run_path}")
                        # Initialize empty training_data to prevent errors
                        training_data = {"epoch": [], "loss": [], "accuracy": []}
                else:
                    st.warning(f"⚠️ Training results not found. Expected at: {final_run_path if final_run_path else 'runs/segment/train*'}")
                    # Initialize empty training_data to prevent errors
                    training_data = {"epoch": [], "loss": [], "accuracy": []}
                
                # Extract final metrics - use the data we just loaded from the correct training run
                # Calculate metrics from the final_df to ensure consistency
                if final_run_path and (final_run_path / "results.csv").exists() and final_df is not None and len(final_df) > 0:
                    # Re-extract metrics from the final_df to ensure we're using the correct data
                    from utils.yolo_metrics import extract_yolo_metrics
                    yolo_metrics = extract_yolo_metrics(final_df)
                    
                    # Get final loss and accuracy from the last epoch (not best)
                    final_loss_val = training_data.get("loss", [])
                    final_acc_val = training_data.get("accuracy", [])
                    
                    # Calculate best metrics from the full training data
                    best_loss_val = min(final_loss_val) if final_loss_val else 0.0
                    best_acc_val = max(final_acc_val) if final_acc_val else 0.0
                    
                    final_metrics = {
                        "final_loss": final_loss_val[-1] if final_loss_val else yolo_metrics.get("best_loss", 0.0),
                        "final_accuracy": final_acc_val[-1] if final_acc_val else yolo_metrics.get("accuracy", 0.0),
                        "best_loss": best_loss_val if final_loss_val else yolo_metrics.get("best_loss", 0.0),
                        "best_accuracy": best_acc_val if final_acc_val else yolo_metrics.get("accuracy", 0.0)
                    }
                else:
                    # Fallback to training_result metrics if we couldn't load the data
                    yolo_metrics = training_result.get("metrics", {})
                    final_loss_val = training_data.get("loss", [])
                    final_acc_val = training_data.get("accuracy", [])
                    final_metrics = {
                        "final_loss": yolo_metrics.get("best_loss", final_loss_val[-1] if final_loss_val else 0.0),
                        "final_accuracy": yolo_metrics.get("accuracy", final_acc_val[-1] if final_acc_val else 0.0),
                        "best_loss": yolo_metrics.get("best_loss", min(final_loss_val) if final_loss_val else 0.0),
                        "best_accuracy": yolo_metrics.get("accuracy", max(final_acc_val) if final_acc_val else 0.0)
                    }
                
                # Get actual epochs used (from config or YOLO default)
                actual_epochs = config.get("epochs", "default")
                
                # Get the actual weights path from training result
                weights_path = training_result.get("best_weights")
                if weights_path:
                    # Ensure it's a string and relative path
                    weights_path = str(Path(weights_path))
                
                # Get results path to extract args.yaml for default values
                results_path = training_result.get("results_path")
                
                # Save training run with actual weights path and results path
                # results_path will be used to extract default values from args.yaml
                run_id = save_training_run(config, final_metrics, weights_path, results_path)
                st.success(f"✅ Training run saved with ID: {run_id}")
                
                # Store download paths in session state for persistent download buttons
                # Get actual weights path from training result
                weights_path_obj = Path(training_result.get("best_weights", "runs/segment/train/weights/best.pt"))
                
                if weights_path_obj.exists():
                    st.session_state.download_weights_path = str(weights_path_obj)
                else:
                    # Try to find weights in the results path
                    if training_result.get("results_path"):
                        alt_weights = Path(training_result["results_path"]) / "weights" / "best.pt"
                        if alt_weights.exists():
                            st.session_state.download_weights_path = str(alt_weights)
                
                # Generate and store report path
                report_path = generate_training_report(run_id, config, final_metrics, training_data)
                if report_path and Path(report_path).exists():
                    st.session_state.download_report_path = str(report_path)
                
                # Display download options immediately after training completes
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.download_weights_path and Path(st.session_state.download_weights_path).exists():
                        with open(st.session_state.download_weights_path, "rb") as f:
                            weights_data = f.read()
                            st.download_button(
                                "Download Weights (best.pt)",
                                weights_data,
                                file_name="best.pt",
                                mime="application/octet-stream",
                                key="download_weights_btn_immediate"
                            )
                    else:
                        st.info("Weights file not available")
                
                with col2:
                    if st.session_state.download_report_path and Path(st.session_state.download_report_path).exists():
                        with open(st.session_state.download_report_path, "rb") as f:
                            report_data = f.read()
                            file_ext = Path(st.session_state.download_report_path).suffix
                            mime_type = "application/pdf" if file_ext == ".pdf" else "text/html"
                            st.download_button(
                                f"Download Training Report ({file_ext[1:].upper()})",
                                report_data,
                                file_name=Path(st.session_state.download_report_path).name,
                                mime=mime_type,
                                key="download_report_btn_immediate"
                            )
                    else:
                        st.info("Training report not available")
                
                # Rerun to refresh the training history table
                # Chart, logs, and download buttons will persist because they're stored in session state
                st.rerun()
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                st.exception(e)

