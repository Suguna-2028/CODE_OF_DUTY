"""
Model Module - PyTorch model for solar panel segmentation
"""

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
import os


class SolarPanelSegmentationModel(nn.Module):
    """
    Lightweight U-Net with MobileNetV2 backbone for solar panel segmentation.
    Outputs a pixel-wise binary mask.
    """
    
    def __init__(self, pretrained=True):
        super(SolarPanelSegmentationModel, self).__init__()
        
        # MobileNetV2 backbone (encoder)
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.encoder = mobilenet.features
        
        # Decoder (upsampling path)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output probability for each pixel
        )
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        x = self.decoder(features)
        
        # Segmentation head
        mask = self.segmentation_head(x)
        
        return mask


def run_inference(image_data, model_path='../model.pt', threshold=0.5, panel_threshold=0.005):
    """
    Run inference on satellite image to detect solar panels.
    
    Args:
        image_data (bytes or file-like): Raw image data
        model_path (str): Path to trained model file
        threshold (float): Threshold for binary mask (default: 0.5)
        panel_threshold (float): Minimum percentage of panel-covered pixels (default: 0.5%)
    
    Returns:
        dict: Dictionary containing:
            - mask: Binary pixel mask (numpy array)
            - has_solar: Boolean indicating if solar panels detected
            - panel_percentage: Percentage of panel-covered pixels
            - confidence: Confidence score based on prediction certainty
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SolarPanelSegmentationModel(pretrained=False)
    
    # Try to load trained weights
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded model from {model_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load model weights: {e}")
            print("   Using untrained model for demonstration")
    else:
        print(f"⚠️  Warning: Model file '{model_path}' not found. Using untrained model.")
    
    model.to(device)
    model.eval()
    
    # Preprocess image
    img = Image.open(image_data)
    img = img.convert('RGB')
    img = img.resize((640, 640))
    
    # Convert to tensor and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Get probability mask
    prob_mask = output.squeeze().cpu().numpy()
    
    # Create binary mask
    binary_mask = (prob_mask > threshold).astype(np.uint8)
    
    # Calculate panel coverage percentage
    total_pixels = binary_mask.size
    panel_pixels = np.sum(binary_mask)
    panel_percentage = (panel_pixels / total_pixels) * 100
    
    # Determine if solar panels are present
    has_solar = panel_percentage > (panel_threshold * 100)
    
    # Calculate confidence score based on prediction certainty
    # Higher confidence when predictions are closer to 0 or 1 (more certain)
    certainty = np.abs(prob_mask - 0.5) * 2  # Scale to 0-1
    confidence = float(np.mean(certainty))
    
    # Adjust confidence based on panel coverage
    if has_solar:
        # Boost confidence if significant panel area detected
        confidence = min(confidence * (1 + panel_percentage / 100), 1.0)
    
    return {
        'mask': binary_mask,
        'has_solar': has_solar,
        'panel_percentage': float(panel_percentage),
        'confidence': float(confidence)
    }