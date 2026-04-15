import geopandas as gpd
import pandas as pd
import os
import glob

print("=" * 60)
print("MERGING DUSA U-NET + RCNN PREDICTIONS")
print("=" * 60)

UNET_DIR = 'outputs/predictions'
RCNN_DIR = 'outputs/rcnn_utilities'
MERGED_DIR = 'outputs/final_predictions'

os.makedirs(MERGED_DIR, exist_ok=True)

unet_files = glob.glob(f'{UNET_DIR}/*.gpkg')
rcnn_files = glob.glob(f'{RCNN_DIR}/*_utilities.gpkg')

rcnn_map = {}
for f in rcnn_files:
    village = os.path.basename(f).replace('_utilities.gpkg', '')
    rcnn_map[village] = f

print(f"\n📂 U-Net files: {len(unet_files)}")
print(f"📂 RCNN files: {len(rcnn_files)}")

for unet_file in unet_files:
    village = os.path.basename(unet_file).replace('_features.gpkg', '')
    unet_gdf = gpd.read_file(unet_file)
    
    if village in rcnn_map:
        rcnn_gdf = gpd.read_file(rcnn_map[village])
        combined = pd.concat([unet_gdf, rcnn_gdf], ignore_index=True)
        print(f"✅ {village}: U-Net({len(unet_gdf)}) + RCNN({len(rcnn_gdf)}) = {len(combined)}")
    else:
        combined = unet_gdf
        print(f"⚠️ {village}: U-Net only ({len(unet_gdf)})")
    
    output_path = f'{MERGED_DIR}/{village}_complete.gpkg'
    combined.to_file(output_path, driver='GPKG')

print(f"\n✅ Merged GeoPackages saved to: {MERGED_DIR}/")
print("=" * 60)
