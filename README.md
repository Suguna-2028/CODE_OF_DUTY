# PV Detection Pipeline

A complete end-to-end pipeline for detecting and measuring solar panel (photovoltaic) installations from satellite imagery using deep learning.

## Overview

This pipeline automatically:
1. Fetches high-resolution satellite imagery for given coordinates
2. Performs quality checks on images (cloud cover, resolution)
3. Detects solar panels using a U-Net segmentation model
4. Calculates PV area in square meters with buffer zone logic
5. Generates JSON results and visual audit overlays

## Features

- **Automated Image Fetching**: Retrieves satellite imagery from Google Static Maps API
- **Quality Control**: Filters out unusable images (clouds, low resolution, darkness)
- **Deep Learning Detection**: Lightweight U-Net with MobileNetV2 backbone
- **Geospatial Calculations**: Converts pixel masks to real-world area measurements
- **Buffer Zone Logic**: Prioritizes 1200 sq ft zone, then 2400 sq ft zone
- **Output Generation**: JSON results and PNG overlay visualizations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries
- `requests` - API calls for image fetching
- `Pillow` - Image processing
- `numpy` - Numerical operations
- `scikit-image` - Image analysis
- `geopy` - Geospatial utilities
- `torch` - PyTorch deep learning framework
- `tensorflow` - Alternative ML framework (optional)
- `GDAL` - Geospatial data processing
- `rasterio` - Raster data I/O

## Project Structure

```
PV_Detection_Pipeline/
├── main.py                          # Main pipeline script
├── requirements.txt                 # Python dependencies
├── MODEL_CARD.md                    # Model documentation
├── model.pt                         # Trained model weights (add your own)
├── utils/
│   ├── __init__.py
│   ├── image_fetcher.py            # Phase 1: Image fetching & QC
│   ├── model.py                    # Phase 2: Model architecture & inference
│   ├── geospatial_tools.py         # Phase 3: Area calculation
│   └── output_generator.py         # Phase 4: Output generation
├── Prediction files/                # JSON output directory (auto-created)
├── Artefacts/                       # Overlay images directory (auto-created)
└── Environment details/             # Environment configuration (optional)
```

## Usage

### Basic Usage

1. **Set your API key** in `main.py`:
```python
api_key = "YOUR_GOOGLE_MAPS_API_KEY"
```

2. **Add your trained model** file as `model.pt` in the root directory

3. **Run the pipeline**:
```bash
python main.py
```

### Processing Custom Coordinates

Edit the `coordinates` list in `main.py`:

```python
coordinates = [
    (37.7749, -122.4194),  # San Francisco
    (34.0522, -118.2437),  # Los Angeles
    # Add your coordinates here
]
```

### Processing Single Location

```python
from main import process_location

result = process_location(
    lat=37.7749,
    lon=-122.4194,
    api_key="YOUR_API_KEY",
    model_path="model.pt"
)

print(f"PV Area: {result['pv_area_sqm_est']} sq meters")
print(f"Has Solar: {result['has_solar']}")
```

## Output Files

### JSON Results (`Prediction files/`)

Each location generates a JSON file with:

```json
{
  "coordinates": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "pv_area_sqm_est": 45.32,
  "bbox_or_mask": {
    "x_min": 120,
    "y_min": 150,
    "x_max": 280,
    "y_max": 310
  },
  "qc_status": "VERIFIED",
  "has_solar": true,
  "confidence": 0.876,
  "panel_percentage": 2.34,
  "buffer_radius_sqft": 1200,
  "metadata": {
    "source": "Google Static Maps API",
    "capture_date": "N/A",
    "pixel_to_meter_ratio": 0.298
  }
}
```

### Visual Overlays (`Artefacts/`)

PNG images showing:
- Original satellite imagery
- Yellow semi-transparent mask over detected panels
- Red bounding box around panel areas

## Pipeline Phases

### Phase 1: Image Fetching
- Queries Google Static Maps API
- Downloads 640x640 satellite imagery at zoom level 20
- Performs quality checks (resolution, cloud cover, brightness)
- Returns image data and metadata

### Phase 2: Model Inference
- Loads trained U-Net model
- Processes image through neural network
- Generates pixel-wise segmentation mask
- Calculates confidence score and panel coverage percentage

### Phase 3: Area Calculation
- Converts pixel counts to square meters using geodesic formulas
- Applies buffer zone logic (1200 sq ft → 2400 sq ft)
- Accounts for latitude-based distortion
- Returns calibrated area estimate

### Phase 4: Output Generation
- Creates JSON with all required fields
- Generates visual overlay with mask and bounding box
- Saves files to appropriate directories

## Configuration

### API Configuration
Get a Google Static Maps API key from [Google Cloud Console](https://console.cloud.google.com/)

### Model Configuration
- Default model path: `model.pt`
- Supports PyTorch `.pt` or `.pth` files
- Model must match the architecture in `utils/model.py`

### Quality Check Thresholds
Edit in `utils/image_fetcher.py`:
```python
# Minimum resolution
if width < 400 or height < 400:
    # Reject image

# Cloud cover threshold
if avg_brightness > 200:
    # Reject image
```

### Detection Thresholds
Edit in `utils/model.py`:
```python
run_inference(
    image_data,
    threshold=0.5,           # Pixel classification threshold
    panel_threshold=0.005    # Minimum panel coverage (0.5%)
)
```

## Troubleshooting

### Common Issues

**"Model file not found"**
- Ensure `model.pt` exists in the root directory
- Check the `model_path` parameter

**"API request failed"**
- Verify your Google Maps API key is valid
- Check internet connectivity
- Ensure API quota is not exceeded

**"Image quality check failed"**
- Try different coordinates
- Check if location has recent satellite imagery
- Adjust quality thresholds if needed

**Low detection accuracy**
- Verify model is properly trained
- Check image resolution and quality
- Review MODEL_CARD.md for known limitations

## Model Information

See [MODEL_CARD.md](MODEL_CARD.md) for detailed information about:
- Model architecture
- Training data
- Evaluation metrics
- Known limitations
- Retraining guidance

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Support

For issues or questions:
- Check the troubleshooting section
- Review MODEL_CARD.md for model-specific questions
- Open an issue with detailed error information

---

**Version**: 1.0  
**Last Updated**: December 2025
