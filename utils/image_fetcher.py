"""
Image Fetcher Module - Fetches satellite imagery from map APIs
"""

import os
import requests
from PIL import Image
from io import BytesIO


def fetch_rooftop_image(lat, lon, api_key=None):
    """
    Fetch high-resolution satellite image centered at coordinates.
    
    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate
        api_key (str): API key for the map service (uses env var if None)
    
    Returns:
        dict: Dictionary containing:
            - image_data: Raw image bytes
            - source: Image source/provider
            - capture_date: Date image was captured
            - qc_status: Quality check status
            - error: Error message if quality check fails
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        return {
            'image_data': None,
            'source': 'Google Static Maps API',
            'capture_date': None,
            'qc_status': 'NOT_VERIFIABLE',
            'error': 'Invalid or missing API key'
        }
    # Google Static Maps API endpoint
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    # Parameters for high-resolution satellite imagery
    params = {
        'center': f'{lat},{lon}',
        'zoom': 20,  # High zoom for rooftop detail
        'size': '640x640',
        'maptype': 'satellite',
        'key': api_key
    }
    
    try:
        # Fetch the image
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        image_data = response.content
        
        # Perform quality check
        qc_result = _check_image_quality(image_data)
        
        # Build result dictionary
        result = {
            'image_data': image_data,
            'source': 'Google Static Maps API',
            'capture_date': 'N/A',  # Google API doesn't provide this directly
            'qc_status': qc_result['status'],
            'error': qc_result.get('error', None)
        }
        
        return result
        
    except requests.exceptions.RequestException as e:
        return {
            'image_data': None,
            'source': 'Google Static Maps API',
            'capture_date': None,
            'qc_status': 'NOT_VERIFIABLE',
            'error': f'Request failed: {str(e)}'
        }


def _check_image_quality(image_data):
    """
    Check image quality for usability.
    
    Args:
        image_data (bytes): Raw image data
    
    Returns:
        dict: Quality check result with status and optional error
    """
    try:
        # Load image
        img = Image.open(BytesIO(image_data))
        
        # Check 1: Image resolution
        width, height = img.size
        if width < 400 or height < 400:
            return {
                'status': 'NOT_VERIFIABLE',
                'error': 'Low resolution image'
            }
        
        # Check 2: Cloud cover detection (simplified - check for high brightness)
        img_gray = img.convert('L')
        pixels = list(img_gray.getdata())
        avg_brightness = sum(pixels) / len(pixels)
        
        # If average brightness is very high, likely heavy cloud cover
        if avg_brightness > 200:
            return {
                'status': 'NOT_VERIFIABLE',
                'error': 'Heavy cloud cover detected'
            }
        
        # Check 3: Very dark images (night imagery or errors)
        if avg_brightness < 30:
            return {
                'status': 'NOT_VERIFIABLE',
                'error': 'Image too dark'
            }
        
        # Image passes quality checks
        return {'status': 'VERIFIED'}
        
    except Exception as e:
        return {
            'status': 'NOT_VERIFIABLE',
            'error': f'Quality check failed: {str(e)}'
        }
