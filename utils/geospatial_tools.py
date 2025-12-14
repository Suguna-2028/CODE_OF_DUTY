"""
Geospatial Tools Module - Area calculation and buffer logic
"""

import math
import numpy as np


def calculate_area(mask, lat, lon, image_metadata):
    """
    Calculate the estimated PV area in square meters from pixel mask.
    
    Args:
        mask (numpy.ndarray): Binary pixel mask
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate
        image_metadata (dict): Image metadata containing zoom level or scale info
    
    Returns:
        dict: Dictionary containing:
            - pv_area_sqm_est: Estimated PV area in square meters
            - buffer_radius_sqft: Buffer radius used (1200 or 2400 sq ft)
            - pixel_to_meter_ratio: Conversion ratio used
    """
    # Get zoom level from metadata (default to 20 for high-res satellite)
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
    
    # Apply buffer logic to check area around closest structure
    buffer_result = _apply_buffer_logic(pv_area_sqm, mask, area_per_pixel)
    
    return {
        'pv_area_sqm_est': buffer_result['final_area'],
        'buffer_radius_sqft': buffer_result['buffer_radius_sqft'],
        'pixel_to_meter_ratio': meters_per_pixel
    }


def _apply_buffer_logic(initial_area_sqm, mask, area_per_pixel):
    """
    Apply buffer zone logic: prioritize 1200 sq ft zone, then 2400 sq ft zone.
    
    Args:
        initial_area_sqm (float): Initial calculated area
        mask (numpy.ndarray): Binary pixel mask
        area_per_pixel (float): Area per pixel in square meters
    
    Returns:
        dict: Final area and buffer radius used
    """
    # Convert square feet to square meters (1 sq ft = 0.092903 sq m)
    SQFT_TO_SQM = 0.092903
    
    # Define buffer zones in square feet
    BUFFER_1200_SQFT = 1200
    BUFFER_2400_SQFT = 2400
    
    # Convert to square meters
    buffer_1200_sqm = BUFFER_1200_SQFT * SQFT_TO_SQM
    buffer_2400_sqm = BUFFER_2400_SQFT * SQFT_TO_SQM
    
    # Calculate radius for each buffer zone (assuming circular area)
    radius_1200_m = math.sqrt(buffer_1200_sqm / math.pi)
    radius_2400_m = math.sqrt(buffer_2400_sqm / math.pi)
    
    # Get image center (assuming structure is at center)
    height, width = mask.shape
    center_y, center_x = height // 2, width // 2
    
    # Check 1200 sq ft buffer zone first
    buffer_1200_area = _calculate_area_in_buffer(
        mask, center_x, center_y, radius_1200_m, area_per_pixel
    )
    
    if buffer_1200_area > 0:
        # PV found in 1200 sq ft zone
        return {
            'final_area': buffer_1200_area,
            'buffer_radius_sqft': BUFFER_1200_SQFT
        }
    
    # Check 2400 sq ft buffer zone
    buffer_2400_area = _calculate_area_in_buffer(
        mask, center_x, center_y, radius_2400_m, area_per_pixel
    )
    
    if buffer_2400_area > 0:
        # PV found in 2400 sq ft zone
        return {
            'final_area': buffer_2400_area,
            'buffer_radius_sqft': BUFFER_2400_SQFT
        }
    
    # No PV found in either buffer zone, return initial area
    return {
        'final_area': initial_area_sqm,
        'buffer_radius_sqft': BUFFER_2400_SQFT
    }


def _calculate_area_in_buffer(mask, center_x, center_y, radius_m, area_per_pixel):
    """
    Calculate PV area within a circular buffer zone.
    
    Args:
        mask (numpy.ndarray): Binary pixel mask
        center_x (int): Center X coordinate in pixels
        center_y (int): Center Y coordinate in pixels
        radius_m (float): Buffer radius in meters
        area_per_pixel (float): Area per pixel in square meters
    
    Returns:
        float: PV area within buffer zone in square meters
    """
    height, width = mask.shape
    
    # Convert radius from meters to pixels
    meters_per_pixel = math.sqrt(area_per_pixel)
    radius_pixels = radius_m / meters_per_pixel
    
    # Create circular mask for buffer zone
    y_coords, x_coords = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    buffer_mask = distance_from_center <= radius_pixels
    
    # Apply buffer mask to PV mask
    pv_in_buffer = mask & buffer_mask
    
    # Calculate area
    panel_pixels_in_buffer = np.sum(pv_in_buffer)
    area_in_buffer = panel_pixels_in_buffer * area_per_pixel
    
    return area_in_buffer
