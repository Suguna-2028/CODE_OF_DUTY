"""
Output Generator Module - Generate JSON and visual artifacts
"""

import json
import os
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO


def generate_output_files(sample_data):
    """
    Generate output files from pipeline results.
    
    Args:
        sample_data (dict): Dictionary containing all pipeline results:
            - lat, lon: Coordinates
            - image_data: Original image bytes
            - mask: Binary segmentation mask
            - has_solar: Boolean
            - panel_percentage: Float
            - confidence: Float
            - pv_area_sqm_est: Float
            - buffer_radius_sqft: Int
            - qc_status: String
            - source: String
            - capture_date: String
    
    Returns:
        dict: Paths to generated files
    """
    # Create output directories
    os.makedirs('Prediction files', exist_ok=True)
    os.makedirs('Artefacts', exist_ok=True)
    
    # Generate unique filename based on coordinates
    lat = sample_data.get('lat', 0)
    lon = sample_data.get('lon', 0)
    base_filename = f"pv_detection_{lat}_{lon}".replace('.', '_')
    
    # 1. Generate JSON output
    json_output = _create_json_output(sample_data)
    json_path = os.path.join('Prediction files', f'{base_filename}.json')
    
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    # 2. Generate audit overlay image
    overlay_path = _create_audit_overlay(sample_data, base_filename)
    
    return {
        'json_file': json_path,
        'overlay_image': overlay_path
    }


def _create_json_output(sample_data):
    """
    Create the required JSON object with all mandatory fields.
    
    Args:
        sample_data (dict): Pipeline results
    
    Returns:
        dict: JSON-serializable output
    """
    # Extract mask for bounding box calculation
    mask = sample_data.get('mask')
    bbox_or_mask = _calculate_bounding_box(mask) if mask is not None else None
    
    json_output = {
        'coordinates': {
            'latitude': sample_data.get('lat'),
            'longitude': sample_data.get('lon')
        },
        'pv_area_sqm_est': round(sample_data.get('pv_area_sqm_est', 0), 2),
        'bbox_or_mask': bbox_or_mask,
        'qc_status': sample_data.get('qc_status', 'VERIFIED'),
        'has_solar': sample_data.get('has_solar', False),
        'confidence': round(sample_data.get('confidence', 0), 3),
        'panel_percentage': round(sample_data.get('panel_percentage', 0), 2),
        'buffer_radius_sqft': sample_data.get('buffer_radius_sqft'),
        'metadata': {
            'source': sample_data.get('source', 'Unknown'),
            'capture_date': sample_data.get('capture_date', 'N/A'),
            'pixel_to_meter_ratio': sample_data.get('pixel_to_meter_ratio')
        }
    }
    
    return json_output


def _calculate_bounding_box(mask):
    """
    Calculate bounding box from binary mask.
    
    Args:
        mask (numpy.ndarray): Binary mask
    
    Returns:
        dict: Bounding box coordinates or None
    """
    if mask is None or not np.any(mask):
        return None
    
    # Find coordinates where mask is True
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return {
        'x_min': int(x_min),
        'y_min': int(y_min),
        'x_max': int(x_max),
        'y_max': int(y_max)
    }


def _create_audit_overlay(sample_data, base_filename):
    """
    Generate PNG/JPEG audit overlay image with predicted mask/bounding box.
    
    Args:
        sample_data (dict): Pipeline results
        base_filename (str): Base filename for output
    
    Returns:
        str: Path to generated overlay image
    """
    # Load original image
    image_data = sample_data.get('image_data')
    if isinstance(image_data, bytes):
        img = Image.open(BytesIO(image_data))
    else:
        img = Image.open(image_data)
    
    img = img.convert('RGBA')
    
    # Create overlay layer
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Get mask
    mask = sample_data.get('mask')
    
    if mask is not None and np.any(mask):
        # Resize mask to match image size if needed
        if mask.shape != (img.height, img.width):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((img.width, img.height), Image.NEAREST)
            mask = np.array(mask_img) > 0
        
        # Draw semi-transparent mask overlay (yellow for solar panels)
        mask_overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_overlay[mask] = [255, 255, 0, 100]  # Yellow with transparency
        mask_layer = Image.fromarray(mask_overlay, 'RGBA')
        overlay = Image.alpha_composite(overlay, mask_layer)
        
        # Draw bounding box
        bbox = _calculate_bounding_box(mask)
        if bbox:
            draw.rectangle(
                [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']],
                outline=(255, 0, 0, 255),
                width=3
            )
    
    # Composite original image with overlay
    result = Image.alpha_composite(img, overlay)
    result = result.convert('RGB')
    
    # Save overlay image
    overlay_path = os.path.join('Artefacts', f'{base_filename}_overlay.png')
    result.save(overlay_path, 'PNG')
    
    return overlay_path
