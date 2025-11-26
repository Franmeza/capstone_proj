"""
Automatic report generation for training runs (PDF or HTML).
"""

from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import base64
import io

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_training_report(run_id: str, config: dict, metrics: dict, training_data: dict, format: str = "pdf") -> str:
    """
    Generate a training report in PDF or HTML format.
    
    Args:
        run_id: Training run ID
        config: Training configuration
        metrics: Training metrics
        training_data: Training progress data (epochs, loss, accuracy)
        format: Output format ("pdf" or "html")
    
    Returns:
        Path to generated report file
    """
    
    if format.lower() == "pdf":
        return generate_pdf_report(run_id, config, metrics, training_data)
    else:
        return generate_html_report(run_id, config, metrics, training_data)


def generate_pdf_report(run_id: str, config: dict, metrics: dict, training_data: dict) -> str:
    """Generate PDF report using FPDF."""
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Training Report", ln=True, align="C")
    pdf.ln(5)
    
    # Run information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Run ID: {run_id}", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Configuration
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Training Configuration", ln=True)
    pdf.set_font("Arial", "", 10)
    
    config_items = [
        ("Data YAML", config.get("data_yml", "N/A")),
        ("Epochs", config.get("epochs", "N/A")),
        ("Learning Rate", config.get("learning_rate", "N/A")),
        ("Batch Size", config.get("batch_size", "N/A")),
        ("Optimizer", config.get("optimizer", "N/A")),
        ("Image Size", config.get("image_size", "N/A")),
    ]
    
    for key, value in config_items:
        pdf.cell(0, 5, f"{key}: {value}", ln=True)
    
    pdf.ln(5)
    
    # Metrics
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Final Metrics", ln=True)
    pdf.set_font("Arial", "", 10)
    
    metrics_items = [
        ("Final Loss", metrics.get("final_loss", "N/A")),
        ("Final Accuracy", metrics.get("final_accuracy", "N/A")),
        ("Best Loss", metrics.get("best_loss", "N/A")),
        ("Best Accuracy", metrics.get("best_accuracy", "N/A")),
    ]
    
    for key, value in metrics_items:
        if isinstance(value, (int, float)):
            pdf.cell(0, 5, f"{key}: {value:.4f}", ln=True)
        else:
            pdf.cell(0, 5, f"{key}: {value}", ln=True)
    
    # Save PDF
    output_path = REPORTS_DIR / f"report_{run_id}.pdf"
    pdf.output(str(output_path))
    
    return str(output_path)


def generate_html_report(run_id: str, config: dict, metrics: dict, training_data: dict) -> str:
    """Generate HTML report with embedded charts."""
    
    # Create training progress chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=training_data.get("epoch", []),
        y=training_data.get("loss", []),
        name="Loss",
        line=dict(color="#d32f2f", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=training_data.get("epoch", []),
        y=training_data.get("accuracy", []),
        name="Accuracy",
        line=dict(color="#4CAF50", width=2),
        yaxis="y2"
    ))
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis2=dict(title="Accuracy", overlaying="y", side="right"),
        template="plotly_white",
        height=400
    )
    
    chart_html = fig.to_html(include_plotlyjs='cdn', div_id="training-chart")
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Report - {run_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .info-card {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }}
            .info-card strong {{
                color: #2c3e50;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Training Report</h1>
            <p><strong>Run ID:</strong> {run_id}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Training Configuration</h2>
            <div class="info-grid">
                <div class="info-card">
                    <strong>Data YAML:</strong><br>{config.get("data_yml", "N/A")}
                </div>
                <div class="info-card">
                    <strong>Epochs:</strong><br>{config.get("epochs", "N/A")}
                </div>
                <div class="info-card">
                    <strong>Learning Rate:</strong><br>{config.get("learning_rate", "N/A")}
                </div>
                <div class="info-card">
                    <strong>Batch Size:</strong><br>{config.get("batch_size", "N/A")}
                </div>
                <div class="info-card">
                    <strong>Optimizer:</strong><br>{config.get("optimizer", "N/A")}
                </div>
                <div class="info-card">
                    <strong>Image Size:</strong><br>{config.get("image_size", "N/A")}
                </div>
            </div>
            
            <h2>Final Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Final Loss</td>
                    <td>{metrics.get("final_loss", "N/A"):.4f if isinstance(metrics.get("final_loss"), (int, float)) else "N/A"}</td>
                </tr>
                <tr>
                    <td>Final Accuracy</td>
                    <td>{metrics.get("final_accuracy", "N/A"):.4f if isinstance(metrics.get("final_accuracy"), (int, float)) else "N/A"}</td>
                </tr>
                <tr>
                    <td>Best Loss</td>
                    <td>{metrics.get("best_loss", "N/A"):.4f if isinstance(metrics.get("best_loss"), (int, float)) else "N/A"}</td>
                </tr>
                <tr>
                    <td>Best Accuracy</td>
                    <td>{metrics.get("best_accuracy", "N/A"):.4f if isinstance(metrics.get("best_accuracy"), (int, float)) else "N/A"}</td>
                </tr>
            </table>
            
            <h2>Training Progress</h2>
            {chart_html}
            
            <div class="footer">
                <p>Generated by YOLO Corrosion Detection Training System</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML
    output_path = REPORTS_DIR / f"report_{run_id}.html"
    output_path.write_text(html_content, encoding="utf-8")
    
    return str(output_path)

