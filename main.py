"""
PV Detection Pipeline - Main Entry Point
Integrates all phases: image fetching, model inference, area calculation, and output generation.
"""

from io import BytesIO
from utils.image_fetcher import fetch_rooftop_image
from utils.model import run_inference
from utils.geospatial_tools import calculate_area
from utils.output_generator import generate_output_files


def process_location(lat, lon, api_key, model_path='model.pt'):
    """
    Process a single location through the complete pipeline.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        api_key (str): Map API key
        model_path (str): Path to trained model
    
    Returns:
        dict: Complete results or error information
    """
    print(f"\n{'='*60}")
    print(f"Processing location: ({lat}, {lon})")
    print(f"{'='*60}")
    
    # Phase 1: Image Fetching
    print("\n[Phase 1] Fetching satellite image...")
    image_result = fetch_rooftop_image(lat, lon, api_key)
    
    # Check QC status
    if image_result['qc_status'] == 'NOT_VERIFIABLE':
        print(f"❌ Image quality check failed: {image_result['error']}")
        return {
            'status': 'failed',
            'qc_status': 'NOT_VERIFIABLE',
            'error': image_result['error'],
            'lat': lat,
            'lon': lon
        }
    
    print(f"✓ Image fetched from {image_result['source']}")
    
    # Phase 2: Model Inference
    print("\n[Phase 2] Running solar panel detection model...")
    image_bytes = BytesIO(image_result['image_data'])
    inference_result = run_inference(image_bytes, model_path=model_path)
    
    print(f"✓ Detection complete - Solar panels: {inference_result['has_solar']}")
    print(f"  Panel coverage: {inference_result['panel_percentage']:.2f}%")
    print(f"  Confidence: {inference_result['confidence']:.3f}")
    
    # Phase 3: Area Calculation
    print("\n[Phase 3] Calculating PV area...")
    image_metadata = {'zoom_level': 20}
    area_result = calculate_area(
        inference_result['mask'],
        lat,
        lon,
        image_metadata
    )
    
    print(f"✓ Estimated PV area: {area_result['pv_area_sqm_est']:.2f} sq meters")
    print(f"  Buffer radius used: {area_result['buffer_radius_sqft']} sq ft")
    
    # Combine all results
    complete_data = {
        'lat': lat,
        'lon': lon,
        'image_data': image_result['image_data'],
        'mask': inference_result['mask'],
        'has_solar': inference_result['has_solar'],
        'panel_percentage': inference_result['panel_percentage'],
        'confidence': inference_result['confidence'],
        'pv_area_sqm_est': area_result['pv_area_sqm_est'],
        'buffer_radius_sqft': area_result['buffer_radius_sqft'],
        'pixel_to_meter_ratio': area_result['pixel_to_meter_ratio'],
        'qc_status': image_result['qc_status'],
        'source': image_result['source'],
        'capture_date': image_result['capture_date']
    }
    
    # Generate output files
    print("\n[Phase 4] Generating output files...")
    output_files = generate_output_files(complete_data)
    
    print(f"✓ JSON saved to: {output_files['json_file']}")
    print(f"✓ Overlay image saved to: {output_files['overlay_image']}")
    
    complete_data['output_files'] = output_files
    complete_data['status'] = 'success'
    
    return complete_data


def main():
    """
    Main function - processes a list of coordinates through the pipeline.
    """
    # List of (lat, lon) coordinates to process
    coordinates = [
        (37.7749, -122.4194),  # San Francisco
        (34.0522, -118.2437),  # Los Angeles
        (40.7128, -74.0060),   # New York
    ]
    
    # API key for map service
    api_key = "YOUR_API_KEY_HERE"
    
    # Model path
    model_path = "model.pt"
    
    print("=" * 60)
    print("PV DETECTION PIPELINE")
    print("=" * 60)
    print(f"Processing {len(coordinates)} locations...")
    
    results = []
    success_count = 0
    failed_count = 0
    
    # Process each location
    for lat, lon in coordinates:
        try:
            result = process_location(lat, lon, api_key, model_path)
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"\n❌ Error processing ({lat}, {lon}): {str(e)}")
            failed_count += 1
            results.append({
                'status': 'error',
                'lat': lat,
                'lon': lon,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total locations: {len(coordinates)}")
    print(f"✓ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print("\nOutput files saved in:")
    print("  - Prediction files/ (JSON results)")
    print("  - Artefacts/ (Overlay images)")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
