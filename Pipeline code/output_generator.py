"""
Output Generator Module - Generate JSON and visual artifacts
"""

import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO


def generate_final_json(data):
    """
    Generate the final JSON object with all 8 mandatory fields.
    
    Args:
        data (dict): Complete pipeline results
    
    Returns:
        str: Path to saved JSON file
    """
    # Create JSON with all mandatory fields
    json_output = {
        'sample_id': data['sample_id'],
        'lat': data['lat'],
        'lon': data['lon'],
        'has_solar': data['has_solar'],
        'confidence': round(data['confidence'], 3),
        'pv_area_sqm_est': round(data['pv_area_sqm_est'], 2),
        'buffer_radius_sqft': data['buffer_radius_sqft'],
        'qc_status': data['qc_status'],
        'bbox_or_mask': data['bbox_or_mask'],
        'image_metadata': data['image_metadata']
    }
    
    # Save to Prediction files folder
    filename = f"{data['sample_id']}_prediction.json"
    filepath = os.path.join('../Prediction files', filename)
    
    # Ensure directory exists
    os.makedirs('../Prediction files', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    return filepath


def generate_audit_overlay(data):
    """
    Generate human-readable PNG overlay with predicted mask/bounding boxes.
    
    Args:
        data (dict): Complete pipeline results
    
    Returns:
        str: Path to saved overlay image
    """
    # Load original image
    image_data = data['image_data']
    img = Image.open(BytesIO(image_data))
    img = img.convert('RGB')
    
    # Convert to OpenCV format for drawing
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Get mask
    mask = data['mask']
    
    if mask is not None and np.any(mask):
        # Resize mask to match image size if needed
        if mask.shape != (img.height, img.width):
            mask = cv2.resize(mask.astype(np.uint8), (img.width, img.height), 
                            interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
        
        # Create colored overlay for solar panels (yellow)
        overlay = img_cv.copy()
        overlay[mask] = [0, 255, 255]  # Yellow in BGR
        
        # Blend with original image (semi-transparent)
        alpha = 0.4
        img_cv = cv2.addWeighted(img_cv, 1-alpha, overlay, alpha, 0)
        
        # Draw bounding box if available
        bbox_str = data['bbox_or_mask']
        if bbox_str:
            try:
                # Parse bounding box string: "(x_min,y_min),(x_max,y_max)"
                coords = bbox_str.replace('(', '').replace(')', '').split(',')
                if len(coords) == 4:
                    x_min, y_min, x_max, y_max = map(int, coords)
                    cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), 
                                (0, 0, 255), 3)  # Red rectangle
            except:
                pass  # Skip if bounding box parsing fails
    
    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Sample ID
    cv2.putText(img_cv, f"ID: {data['sample_id']}", (10, 30), 
                font, font_scale, (255, 255, 255), thickness)
    
    # Solar detection status
    status_text = f"Solar: {'YES' if data['has_solar'] else 'NO'}"
    color = (0, 255, 0) if data['has_solar'] else (0, 0, 255)  # Green/Red
    cv2.putText(img_cv, status_text, (10, 60), 
                font, font_scale, color, thickness)
    
    # Area and confidence
    cv2.putText(img_cv, f"Area: {data['pv_area_sqm_est']:.1f} m²", (10, 90), 
                font, font_scale, (255, 255, 255), thickness)
    cv2.putText(img_cv, f"Conf: {data['confidence']:.2f}", (10, 120), 
                font, font_scale, (255, 255, 255), thickness)
    
    # Buffer zone used
    cv2.putText(img_cv, f"Buffer: {data['buffer_radius_sqft']} sq ft", (10, 150), 
                font, font_scale, (255, 255, 255), thickness)
    
    # QC Status
    qc_color = (0, 255, 0) if data['qc_status'] == 'VERIFIABLE' else (0, 0, 255)
    cv2.putText(img_cv, f"QC: {data['qc_status']}", (10, 180), 
                font, font_scale, qc_color, thickness)
    
    # Save overlay image
    filename = f"{data['sample_id']}_overlay.png"
    filepath = os.path.join('../Artefacts', filename)
    
    # Ensure directory exists
    os.makedirs('../Artefacts', exist_ok=True)
    
    # Convert back to RGB and save
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(filepath, 'PNG')
    
    return filepath