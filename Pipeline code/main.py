"""
PV Detection Pipeline - Main Entry Point
Reads coordinates from Excel file and processes through complete pipeline
"""

import os
import sys
import pandas as pd
from io import BytesIO
import traceback

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_fetcher import fetch_rooftop_image
from model import run_inference
from geospatial_tools import calculate_area, apply_buffer_logic, determine_qc_status
from output_generator import generate_final_json, generate_audit_overlay


def read_input_coordinates(excel_path):
    """
    Read coordinates from Excel file.
    
    Args:
        excel_path (str): Path to Excel file with columns: sample_id, latitude, longitude
    
    Returns:
        pandas.DataFrame: DataFrame with coordinate data
    """
    try:
        df = pd.read_excel(excel_path)
        required_columns = ['sample_id', 'latitude', 'longitude']
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None


def process_single_location(row, api_key, model_path='../model.pt'):
    """
    Process a single location through the complete pipeline.
    
    Args:
        row (pandas.Series): Row with sample_id, latitude, longitude
        api_key (str): Map API key
        model_path (str): Path to trained model
    
    Returns:
        dict: Complete results or error information
    """
    sample_id = row['sample_id']
    lat = row['latitude']
    lon = row['longitude']
    
    print(f"\n{'='*60}")
    print(f"Processing {sample_id}: ({lat}, {lon})")
    print(f"{'='*60}")
    
    try:
        # Phase 1: Image Fetching
        print("\n[Phase 1] Fetching satellite image...")
        image_result = fetch_rooftop_image(lat, lon, api_key)
        
        # Initial QC check
        if image_result['qc_status'] == 'NOT_VERIFIABLE':
            print(f"❌ Image quality check failed: {image_result['error']}")
            return {
                'sample_id': sample_id,
                'lat': lat,
                'lon': lon,
                'has_solar': False,
                'confidence': 0.0,
                'pv_area_sqm_est': 0.0,
                'buffer_radius_sqft': 2400,
                'qc_status': 'NOT_VERIFIABLE',
                'bbox_or_mask': None,
                'image_metadata': {
                    'source': image_result.get('source', 'Unknown'),
                    'capture_date': image_result.get('capture_date', 'N/A'),
                    'error': image_result['error']
                },
                'status': 'failed'
            }
        
        print(f"✓ Image fetched from {image_result['source']}")
        
        # Phase 2: Model Inference
        print("\n[Phase 2] Running solar panel detection model...")
        image_bytes = BytesIO(image_result['image_data'])
        inference_result = run_inference(image_bytes, model_path=model_path)
        
        print(f"✓ Detection complete - Solar panels: {inference_result['has_solar']}")
        print(f"  Panel coverage: {inference_result['panel_percentage']:.2f}%")
        print(f"  Confidence: {inference_result['confidence']:.3f}")
        
        # Phase 3: Buffer Zone Logic and Area Calculation
        print("\n[Phase 3] Applying buffer zone logic and calculating area...")
        
        # Apply buffer zone logic (1200 sq ft -> 2400 sq ft)
        buffer_result = apply_buffer_logic(
            inference_result['mask'],
            lat, lon,
            {'zoom_level': 20}
        )
        
        # Calculate area in square meters
        area_result = calculate_area(
            buffer_result['final_mask'],
            lat, lon,
            {'zoom_level': 20}
        )
        
        print(f"✓ Buffer zone: {buffer_result['buffer_radius_sqft']} sq ft")
        print(f"✓ Estimated PV area: {area_result['pv_area_sqm_est']:.2f} sq meters")
        
        # Final QC determination
        final_qc_status = determine_qc_status(
            image_result,
            inference_result,
            buffer_result
        )
        
        # Combine all results
        complete_data = {
            'sample_id': sample_id,
            'lat': lat,
            'lon': lon,
            'has_solar': buffer_result['has_solar'],
            'confidence': inference_result['confidence'],
            'pv_area_sqm_est': area_result['pv_area_sqm_est'],
            'buffer_radius_sqft': buffer_result['buffer_radius_sqft'],
            'qc_status': final_qc_status,
            'bbox_or_mask': buffer_result['bbox_or_mask'],
            'image_metadata': {
                'source': image_result['source'],
                'capture_date': image_result['capture_date'],
                'pixel_to_meter_ratio': area_result['pixel_to_meter_ratio']
            },
            'image_data': image_result['image_data'],
            'mask': buffer_result['final_mask'],
            'status': 'success'
        }
        
        # Phase 4: Generate outputs
        print("\n[Phase 4] Generating output files...")
        
        # Generate JSON
        json_path = generate_final_json(complete_data)
        print(f"✓ JSON saved to: {json_path}")
        
        # Generate audit overlay
        overlay_path = generate_audit_overlay(complete_data)
        print(f"✓ Overlay image saved to: {overlay_path}")
        
        complete_data['output_files'] = {
            'json_file': json_path,
            'overlay_image': overlay_path
        }
        
        return complete_data
        
    except Exception as e:
        print(f"\n❌ Error processing {sample_id}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'sample_id': sample_id,
            'lat': lat,
            'lon': lon,
            'has_solar': False,
            'confidence': 0.0,
            'pv_area_sqm_est': 0.0,
            'buffer_radius_sqft': 2400,
            'qc_status': 'NOT_VERIFIABLE',
            'bbox_or_mask': None,
            'image_metadata': {
                'source': 'Unknown',
                'capture_date': 'N/A',
                'error': str(e)
            },
            'status': 'error'
        }


def main():
    """
    Main function - processes coordinates from Excel file through the pipeline.
    """
    # Configuration
    excel_path = "input_coordinates.xlsx"  # Input file path
    api_key = os.getenv('GOOGLE_MAPS_API_KEY', 'YOUR_ACTUAL_API_KEY_HERE')
    model_path = "../model.pt"
    
    print("=" * 60)
    print("PV DETECTION PIPELINE")
    print("=" * 60)
    
    # Check if API key is set
    if api_key == 'YOUR_API_KEY_HERE':
        print("⚠️  Warning: Please set GOOGLE_MAPS_API_KEY environment variable")
        print("   or update the api_key variable in main.py")
    
    # Read input coordinates
    print(f"Reading coordinates from: {excel_path}")
    df = read_input_coordinates(excel_path)
    
    if df is None:
        print("❌ Failed to read input file. Creating sample file...")
        # Create sample input file
        sample_data = {
            'sample_id': ['SITE_001', 'SITE_002', 'SITE_003'],
            'latitude': [37.7749, 34.0522, 40.7128],
            'longitude': [-122.4194, -118.2437, -74.0060]
        }
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_excel(excel_path, index=False)
        print(f"✓ Created sample file: {excel_path}")
        df = sample_df
    
    print(f"Processing {len(df)} locations...")
    
    # Create output directories
    os.makedirs('../Prediction files', exist_ok=True)
    os.makedirs('../Artefacts', exist_ok=True)
    
    results = []
    success_count = 0
    failed_count = 0
    
    # Process each location
    for index, row in df.iterrows():
        result = process_single_location(row, api_key, model_path)
        results.append(result)
        
        if result['status'] == 'success':
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total locations: {len(df)}")
    print(f"✓ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print("\nOutput files saved in:")
    print("  - ../Prediction files/ (JSON results)")
    print("  - ../Artefacts/ (Overlay images)")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()