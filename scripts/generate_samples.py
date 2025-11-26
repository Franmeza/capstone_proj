"""
Script to generate sample images for dashboard preview.
Run this script to create dummy sample images in data/samples/
"""

from pathlib import Path
import numpy as np
from PIL import Image

def generate_sample_images():
    """Generate sample images for dashboard preview."""
    
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample1.jpg - Simulated metal surface with rust
    img1 = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    # Add some rust-colored patches
    for _ in range(3):
        center_x = np.random.randint(100, 540)
        center_y = np.random.randint(100, 380)
        radius = np.random.randint(30, 80)
        y, x = np.ogrid[:480, :640]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img1[mask] = [100, 50, 30]  # Rust color
    
    Image.fromarray(img1).save(samples_dir / "sample1.jpg", quality=95)
    print(f"Generated {samples_dir / 'sample1.jpg'}")
    
    # Generate sample2.jpg - Simulated clean metal surface
    img2 = np.random.randint(150, 220, (480, 640, 3), dtype=np.uint8)
    # Add some texture
    noise = np.random.normal(0, 10, (480, 640, 3))
    img2 = np.clip(img2 + noise, 0, 255).astype(np.uint8)
    
    Image.fromarray(img2).save(samples_dir / "sample2.jpg", quality=95)
    print(f"Generated {samples_dir / 'sample2.jpg'}")
    
    print(f"\nSample images generated in {samples_dir}")
    print("You can now run the Streamlit app to see them in the dashboard!")

if __name__ == "__main__":
    generate_sample_images()

