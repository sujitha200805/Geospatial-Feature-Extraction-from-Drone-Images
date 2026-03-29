
SVAMITVA Scheme - AI-Based Feature Extraction from Drone Images
This project uses DuSA U-Net (Dual Self-Attention U-Net) model to automatically extract features from drone orthophotos . The model identifies and classifies:

- Buildings (4 roof types: RCC, Tiled, Tin, Thatched)
- Road Networks
- Water Bodies
- Infrastructure (Transformers, Tanks, Wells)

Achievements
Validation Accuracy- 97.88%
Training Patches- 36,017 
Training Villages- 9 
Feature Classes-10 (0-9) 

Model architecture:

![WhatsApp Image 2026-03-29 at 22 56 56](https://github.com/user-attachments/assets/ed30cb90-0da7-47f2-84da-2d2b9407e0e7)# Geospatial-Feature-Extraction-from-Drone-Images


Tech Stack
- Framework: PyTorch 2.1.0
- Architecture: DuSA U-Net (31.18M params)
- GPU: NVIDIA H200 (8x)
- Geospatial: rasterio, geopandas, shapely
- Augmentation: Albumentations
