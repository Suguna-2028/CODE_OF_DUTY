"""
Geospatial Tools Module - Buffer logic, area calculation, and QC determination
"""

import math
import numpy as np


def apply_buffer_logic(mask, lat, lon, image_metadata):
    """
    Apply mandatory buffer zone logic: 1200 sq ft first, then 2400 sq ft.
    
    Args:
        mask (numpy.ndarray): Binary pixel mask from model
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate
        image_metadata (dict): Image metadata containing zoom level
    
    Returns:
        dict: Dictionary containing:
            - has_solar: Boolean (True if PV found in either buffer)
            - buffer_radius_sqft: Int (1200 or 2400)
            - final_mask: Binary mask within the used buffer
            - bbox_or_mask: Bounding box or mask encoding
    """
    # Convert square feet to square meters
    SQFT_TO_SQM = 0.092903
    
    # Define buffer zones in square feet
    BUFFER_1200_SQFT = 1200
    BUFFER_2400_SQFT = 2400
    
    # Convert to square meters
    buffer_1200_sqm = BUFFER_1200_SQFT * SQFT_TO_SQM
    buffer_2400_sqm = BUFFER_2400_SQFT * SQFT_TO_SQM
    
    # Calculate pixel-to-meter ratio
    zoom_level = image_metadata.get('zoom_level', 20)
    meters_per_pixel = (40075017 * math.cos(math.radians(lat))) / (256 * (2 ** zoom_level))
    
    # Get image center (assuming target coordinate is at center)
    height, width = mask.shape
    center_y, center_x = height // 2, width // 2
    
    # Attempt 1: Check 1200 sq ft buffer zone
    print("  Checking 1200 sq ft buffer zone...")
    buffer_1200_result = _check_buffer_zone(
        mask, center_x, center_y, buffer_1200_sqm, meters_per_pixel
    )
    
    if buffer_1200_result['has_solar']:
        print(f"  ✓ PV found in 1200 sq ft zone")
        return {
            'has_solar': True,
            'buffer_radius_sqft': BUFFER_1200_SQFT,
            'final_mask': buffer_1200_result['mask_in_buffer'],
            'bbox_or_mask': _calculate_bounding_box(buffer_1200_result['mask_in_buffer'])
        }
    
    # Attempt 2: Check 2400 sq ft buffer zone
    print("  No PV in 1200 sq ft zone, checking 2400 sq ft zone...")
    buffer_2400_result = _check_buffer_zone(
        mask, center_x, center_y, buffer_2400_sqm, meters_per_pixel
    )
    
    if buffer_2400_result['has_solar']:
        print(f"  ✓ PV found in 2400 sq ft zone")
        return {
            'has_solar': True,
            'buffer_radius_sqft': BUFFER_2400_SQFT,
            'final_mask': buffer_2400_result['mask_in_buffer'],
            'bbox_or_mask': _calculate_bounding_box(buffer_2400_result['mask_in_buffer'])
        }
    
    # No PV found in either buffer zone
    print("  No PV found in either buffer zone")
    return {
        'has_solar': False,
        'buffer_radius_sqft': BUFFER_2400_SQFT,
        'final_mask': np.zeros_like(mask),
        'bbox_or_mask': None
    }


def _check_buffer_zone(mask, center_x, center_y, buffer_area_sqm, meters_per_pixel):
    """
    Check for PV within a circular buffer zone.
    
    Args:
        mask (numpy.ndarray): Binary pixel mask
        center_x (int): Center X coordinate in pixels
        center_y (int): Center Y coordinate in pixels
        buffer_area_sqm (float): Buffer area in square meters
        meters_per_pixel (float): Meters per pixel conversion
    
    Returns:
        dict: Buffer zone check result
    """
    height, width = mask.shape
    
    # Calculate radius from area (assuming circular buffer)
    radius_m = math.sqrt(buffer_area_sqm / math.pi)
    radius_pixels = radius_m / meters_per_pixel
    
    # Create circular mask for buffer zone
    y_coords, x_coords = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    buffer_mask = distance_from_center <= radius_pixels
    
    # Apply buffer mask to PV mask
    mask_in_buffer = mask & buffer_mask
    
    # Check if any PV pixels exist in buffer
    has_solar = np.any(mask_in_buffer)
    
    return {
        'has_solar': has_solar,
        'mask_in_buffer': mask_in_buffer
    }


def calculate_area(mask, lat, lon, image_metadata):
    """
    Convert pixel count to square meters using geospatial metadata.
    
    Args:
        mask (numpy.ndarray): Binary pixel mask
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate
        image_metadata (dict): Image metadata containing zoom level
    
    Returns:
        dict: Dictionary containing:
            - pv_area_sqm_est: Estimated PV area in square meters
            - pixel_to_meter_ratio: Conversion ratio used
    """
    # Get zoom level from metadata
    zoom_level = image_metadata.get('zoom_level', 20)
    
    # Calculate pixel-to-meter ratio using zoom level and latitude
    # At zoom level z, the map width is 256 * 2^z pixels
    # Earth's circumference at equator is ~40,075,017 meters
    # Adjust for latitude using cosine
    meters_per_pixel = (40075017 * math.cos(math.radians(lat))) / (256 * (2 ** zoom_level))
    
    # Count panel pixels in mask
    panel_pixels = np.sum(mask)
    
    # Calculate area per pixel (square meters)
    area_per_pixel = meters_per_pixel ** 2
    
    # Calculate total PV area
    pv_area_sqm = panel_pixels * area_per_pixel
    
    return {
        'pv_area_sqm_est': float(pv_area_sqm),
        'pixel_to_meter_ratio': float(meters_per_pixel)
    }


def determine_qc_status(image_result, inference_result, buffer_result):
    """
    Determine final QC status based on image quality and model confidence.
    
    Args:
        image_result (dict): Results from image fetching
        inference_result (dict): Results from model inference
        buffer_result (dict): Results from buffer zone logic
    
    Returns:
        str: 'VERIFIABLE' or 'NOT_VERIFIABLE'
    """
    # Check image fetch status
    if image_result['qc_status'] == 'NOT_VERIFIABLE':
        return 'NOT_VERIFIABLE'
    
    # Check model confidence for detected areas
    if buffer_result['has_solar']:
        # If solar panels detected, check confidence
        if inference_result['confidence'] < 0.3:
            return 'NOT_VERIFIABLE'  # Low confidence on detection
    
    # Check for potential heavy occlusion (very low panel percentage with detection)
    if (buffer_result['has_solar'] and 
        inference_result['panel_percentage'] < 0.1 and 
        inference_result['confidence'] < 0.5):
        return 'NOT_VERIFIABLE'  # Likely heavily occluded
    
    return 'VERIFIABLE'


def _calculate_bounding_box(mask):
    """
    Calculate bounding box from binary mask.
    
    Args:
        mask (numpy.ndarray): Binary mask
    
    Returns:
        str or None: Bounding box string or None if no mask
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
    
    # Return as string format for JSON
    return f"({x_min},{y_min}),({x_max},{y_max})"