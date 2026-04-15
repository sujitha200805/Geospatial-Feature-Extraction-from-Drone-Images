
"""
Run Faster R-CNN on testing villages using patch-based processing
to avoid GPU memory issues
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
import glob
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

print("=" * 60)
print("FASTER R-CNN UTILITY DETECTION (PATCH-BASED)")
print("=" * 60)

# ============================================
# MODEL DEFINITION
# ============================================
def get_model(num_classes=4):
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    
    model = fasterrcnn_resnet50_fpn(
        weights='DEFAULT',
        rpn_anchor_generator=anchor_generator,
        box_nms_thresh=0.2
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# ============================================
# LOAD MODEL
# ============================================
print("\n📂 Loading Faster R-CNN model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=4).to(device)

model_path = 'outputs/final/faster_rcnn_utilities.pth'
if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Model loaded")

CLASS_MAPPING = {1: 'Transformer', 2: 'Tank', 3: 'Well'}

# ============================================
# PATCH-BASED DETECTION
# ============================================
def detect_utilities_patched(image_path, patch_size=1024, stride=512, confidence_threshold=0.3):
    """
    Process large orthophoto in patches to avoid GPU memory issues
    """
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        
        print(f"   Image size: {width} x {height}")
        print(f"   Processing in {patch_size}x{patch_size} patches...")
        
        all_detections = []
        patch_count = 0
        total_patches = ((height - patch_size) // stride + 1) * ((width - patch_size) // stride + 1)
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                window = Window(x, y, patch_size, patch_size)
                patch = src.read(window=window)
                
                # Skip if patch is mostly empty (optional optimization)
                if patch.shape[0] >= 3:
                    patch_rgb = patch[:3]  # Take RGB bands
                    patch_rgb = np.transpose(patch_rgb, (1, 2, 0))
                    
                    # Convert to tensor
                    img_tensor = torch.from_numpy(patch_rgb.astype(np.float32) / 255.0)
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        predictions = model(img_tensor)
                    
                    # Extract detections in this patch
                    boxes = predictions[0]['boxes'].cpu().numpy()
                    labels = predictions[0]['labels'].cpu().numpy()
                    scores = predictions[0]['scores'].cpu().numpy()
                    
                    for box, label, score in zip(boxes, labels, scores):
                        if score >= confidence_threshold and label in CLASS_MAPPING:
                            # Convert patch coordinates to global coordinates
                            x1, y1, x2, y2 = box
                            cx = (x1 + x2) / 2 + x
                            cy = (y1 + y2) / 2 + y
                            
                            # Convert to geographic coordinates
                            lon, lat = transform * (cx, cy)
                            
                            all_detections.append({
                                'geometry': Point(lon, lat),
                                'class_name': CLASS_MAPPING[label],
                                'confidence': float(score)
                            })
                
                patch_count += 1
                if patch_count % 50 == 0:
                    print(f"      Patches: {patch_count}/{total_patches}")
                
                # Clear GPU cache periodically
                if patch_count % 100 == 0:
                    torch.cuda.empty_cache()
    
    if all_detections:
        return gpd.GeoDataFrame(all_detections, crs=crs)
    else:
        return gpd.GeoDataFrame(columns=['geometry', 'class_name', 'confidence'], crs=crs)

# ============================================
# PROCESS TESTING VILLAGES
# ============================================
test_images = sorted(glob.glob('data/testing/images/*.tif'))
print(f"\n📸 Found {len(test_images)} testing villages")

os.makedirs('outputs/rcnn_utilities', exist_ok=True)

results = []

for img_path in tqdm(test_images, desc="Processing villages"):
    base = os.path.splitext(os.path.basename(img_path))[0]
    output_path = f'outputs/rcnn_utilities/{base}_utilities.gpkg'
    
    print(f"\n📸 {base}")
    
    try:
        gdf = detect_utilities_patched(img_path, patch_size=1024, stride=512, confidence_threshold=0.3)
        gdf.to_file(output_path, driver='GPKG')
        
        results.append({
            'Village': base,
            'Utilities': len(gdf),
            'Transformers': len(gdf[gdf['class_name'] == 'Transformer']) if len(gdf) > 0 else 0,
            'Tanks': len(gdf[gdf['class_name'] == 'Tank']) if len(gdf) > 0 else 0,
            'Wells': len(gdf[gdf['class_name'] == 'Well']) if len(gdf) > 0 else 0
        })
        
        print(f"   ✅ Detected {len(gdf)} utilities")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append({'Village': base, 'Utilities': 0, 'Transformers': 0, 'Tanks': 0, 'Wells': 0})
    
    torch.cuda.empty_cache()

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("📊 RCNN UTILITY DETECTION SUMMARY")
print("=" * 60)

print(f"\n{'Village':<40} {'Utilities':<10} {'Trans':<8} {'Tank':<8} {'Well':<8}")
print("-" * 74)
for r in results:
    print(f"{r['Village']:<40} {r['Utilities']:<10} {r['Transformers']:<8} {r['Tanks']:<8} {r['Wells']:<8}")

total_utils = sum(r['Utilities'] for r in results)
print(f"\n✅ Total utilities detected: {total_utils}")
print(f"📁 Output saved to: outputs/rcnn_utilities/")
print("=" * 60)

import pandas as pd
pd.DataFrame(results).to_csv('outputs/rcnn_utilities/summary.csv', index=False)
