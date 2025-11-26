# YOLO Corrosion Detection UI

A modern, minimal Streamlit UI for a computer vision project using YOLO segmentation to detect corrosion in metals.

## Features

- **Dashboard**: View model metrics, training progress charts, and sample predictions
- **Prediction**: Upload images or videos to detect corrosion with segmentation overlays
- **Training**: Configure hyperparameters and train models with live progress tracking
- **Training History**: Track and compare past training runs
- **Automatic Reports**: Generate PDF or HTML reports after training

## Project Structure

```
/app.py                        # Main Streamlit app entry point
/components/                   
    ├── dashboard.py            # Dashboard page
    ├── prediction.py           # Prediction page
    └── training.py             # Training page
/utils/                        
    ├── model.py                # Placeholder YOLO inference and training functions
    ├── history.py              # Model training history handler (SQLite)
    └── report.py               # Automatic report generation (PDF or HTML)
/data/                         
    ├── example_data.yml        # Example dataset config
    └── samples/                # Folder for sample images/videos
/scripts/
    └── generate_samples.py     # Script to generate sample images
/requirements.txt               # Project dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate sample images (optional):
```bash
python scripts/generate_samples.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

### Dashboard
- View model performance metrics (accuracy, precision, recall, IoU, mAP)
- Explore training progress charts
- Preview sample predictions

### Prediction
- Upload images or videos via file upload or path input
- Run inference to detect corrosion
- Download processed results with segmentation overlays

### Training
- Configure training parameters (epochs, learning rate, batch size, optimizer, etc.)
- Start training with live progress tracking
- Download trained weights and training reports
- Compare different training runs

## Notes

- This is a placeholder UI ready for YOLO integration
- All model functions in `utils/model.py` are placeholders
- Replace placeholder functions with actual YOLO logic when ready
- Training history is stored in SQLite database at `data/training_history.db`
- Reports are saved in the `reports/` directory

## Future Integration

To integrate with actual YOLO models:

1. Replace placeholder functions in `utils/model.py` with real YOLO inference
2. Update `train_model()` function with actual YOLO training logic
3. Modify `predict_image_or_video()` to use your trained YOLO model
4. Update data paths and configurations as needed

