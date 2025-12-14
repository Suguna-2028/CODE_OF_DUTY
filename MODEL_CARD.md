# Model Card: PV Detection Pipeline

## Model Overview
This model performs semantic segmentation of solar panels (photovoltaic systems) from high-resolution satellite imagery using a lightweight U-Net architecture with a MobileNetV2 backbone.

---

## Data Used

### Training Data
- **Source**: High-resolution satellite imagery from Google Static Maps API and similar providers
- **Resolution**: 640x640 pixels at zoom level 20 (~0.3m per pixel)
- **Coverage**: Rooftop solar installations across various geographic regions
- **Annotations**: Pixel-wise binary masks indicating solar panel locations
- **Data Split**: 
  - Training: 70%
  - Validation: 15%
  - Test: 15%

### Data Characteristics
- **Image Types**: RGB satellite imagery
- **Conditions**: Various lighting conditions, seasons, and roof types
- **Geographic Diversity**: Urban, suburban, and rural areas
- **Panel Types**: Various solar panel configurations (grid-tied residential/commercial)

---

## Model Architecture

### U-Net with MobileNetV2 Backbone

**Encoder (MobileNetV2)**:
- Pretrained on ImageNet for transfer learning
- Lightweight architecture optimized for mobile/edge deployment
- Extracts hierarchical features from input images
- Output: 1280 feature channels

**Decoder (Upsampling Path)**:
- 5 transposed convolution layers
- Progressive upsampling: 1280 → 512 → 256 → 128 → 64 → 32 channels
- ReLU activation between layers
- Restores spatial resolution to match input size

**Segmentation Head**:
- Two convolutional layers (32 → 16 → 1 channels)
- Sigmoid activation for pixel-wise probability output
- Binary classification per pixel (solar panel vs. background)

**Model Parameters**: ~3.5M (lightweight for deployment)

---

## Evaluation Metrics

### Primary Metrics
- **F1 Score**: Harmonic mean of precision and recall
  - Target: > 0.85 on test set
  - Balances false positives and false negatives

- **RMSE (Root Mean Square Error)**: For area estimation accuracy
  - Measures deviation between predicted and actual PV area
  - Target: < 5 square meters for typical residential installations

### Additional Metrics
- **IoU (Intersection over Union)**: Measures mask overlap quality
- **Precision**: Ratio of true positive pixels to all predicted positive pixels
- **Recall**: Ratio of true positive pixels to all actual positive pixels
- **Pixel Accuracy**: Overall percentage of correctly classified pixels

---

## Known Limitations

### Environmental Factors
1. **Shadow Occlusion**: 
   - Heavy shadows from trees, buildings, or clouds can obscure panels
   - May result in underestimation of PV area
   - Mitigation: Multi-temporal imagery analysis (future work)

2. **Tree Occlusion**:
   - Overhanging vegetation can hide solar panels
   - Partial occlusion may lead to fragmented detections
   - Mitigation: Seasonal imagery when trees have fewer leaves

3. **Cloud Cover**:
   - Heavy cloud cover triggers quality check failure
   - Thin clouds may reduce detection confidence
   - Mitigation: Automatic retry with different imagery dates

### Technical Limitations
4. **Image Resolution**:
   - Performance degrades below 0.5m per pixel resolution
   - Small residential installations may be missed at lower resolutions

5. **Panel Orientation**:
   - Flat/horizontal panels may be confused with skylights or roof features
   - Very dark panels in shadow may blend with roof material

6. **Geographic Bias**:
   - Model trained primarily on North American installations
   - May have reduced accuracy in regions with different roof/panel styles

7. **Temporal Accuracy**:
   - Satellite imagery may be outdated
   - Recent installations or removals may not be reflected

---

## Retraining Guidance

### When to Retrain
- Model F1 score drops below 0.80 on validation set
- Expanding to new geographic regions with different characteristics
- New solar panel types or installation patterns emerge
- Significant changes in satellite imagery quality/resolution

### Retraining Process
1. **Data Collection**:
   - Gather new labeled examples (minimum 500 images)
   - Ensure geographic and condition diversity
   - Include challenging cases (shadows, occlusions, edge cases)

2. **Data Augmentation**:
   - Random rotations (0°, 90°, 180°, 270°)
   - Horizontal/vertical flips
   - Brightness and contrast adjustments
   - Gaussian noise addition

3. **Training Configuration**:
   - Optimizer: Adam (learning rate: 1e-4)
   - Loss function: Binary Cross-Entropy with Dice Loss
   - Batch size: 16
   - Epochs: 50-100 with early stopping
   - Learning rate scheduler: ReduceLROnPlateau

4. **Validation**:
   - Monitor F1 score and IoU on validation set
   - Perform error analysis on failure cases
   - Test on held-out geographic regions

5. **Deployment**:
   - A/B test new model against current production model
   - Gradual rollout with monitoring
   - Maintain model versioning and rollback capability

### Fine-tuning for Specific Regions
- Use pretrained weights as initialization
- Train on region-specific data for 10-20 epochs
- Lower learning rate (1e-5) to preserve general features

---

## Model Maintenance
- **Monitoring**: Track prediction confidence and QC failure rates
- **Feedback Loop**: Collect user corrections for continuous improvement
- **Version Control**: Maintain model registry with performance metrics
- **Documentation**: Update this card with each model version

---

**Model Version**: 1.0  
**Last Updated**: December 2025  
**Contact**: PV Detection Team
