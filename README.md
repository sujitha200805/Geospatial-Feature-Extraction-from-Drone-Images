![WhatsApp Image 2026-03-29 at 22 56 56](https://github.com/user-attachments/assets/ed30cb90-0da7-47f2-84da-2d2b9407e0e7)# Geospatial-Feature-Extraction-from-Drone-Images
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
┌─────────────────────────────────────┐
│ Input (512x512x3)                   │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ DoubleConv (64)                     │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ Down 1 (128)                        │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ Down 2 (256)                        │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ Down 3 (512)                        │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│           Bottleneck (1024)         |
│ ┌─────────────────────────┐         │
│ │ Channel Attention       │         │
│ │ Spatial Attention       │         │
│ └─────────────────────────┘         │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│    Up 1 (512)                       │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  Up 2 (256)                         │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ Up 3 (128)                          │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ Up 4 (64)                           │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ Output (512x512x10)                 │
│ Classes: 0=BG, 1-9=Features         │
└─────────────────────────────────────┘


Tech Stack
- Framework: PyTorch 2.1.0
- Architecture: DuSA U-Net (31.18M params)
- GPU: NVIDIA H200 (8x)
- Geospatial: rasterio, geopandas, shapely
- Augmentation: Albumentations
