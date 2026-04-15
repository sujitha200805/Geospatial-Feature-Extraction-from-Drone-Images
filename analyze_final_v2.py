"""
Complete analysis of final GeoPackages and model performance
(Fixed for all classes)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 FINAL GEOPACKAGE ANALYSIS & MODEL PERFORMANCE")
print("=" * 80)

# ============================================
# 1. LOAD ALL FINAL GEOPACKAGES
# ============================================
FINAL_DIR = 'outputs/final_predictions'
if not os.path.exists(FINAL_DIR):
    FINAL_DIR = 'outputs/testing_predictions'

gpkg_files = glob.glob(f'{FINAL_DIR}/*.gpkg')
print(f"\n📂 Found {len(gpkg_files)} GeoPackage files")

if len(gpkg_files) == 0:
    print("❌ No GeoPackage files found! Run merge script first.")
    exit()

# ============================================
# 2. CLASS MAPPING
# ============================================
CLASS_NAMES = {
    1: 'RCC_Roof',
    2: 'Tiled_Roof',
    3: 'Tin_Roof',
    4: 'Thatched_Roof',
    5: 'Road',
    6: 'Waterbody',
    7: 'Transformer',
    8: 'Tank',
    9: 'Well'
}

MODEL_SOURCES = {
    'RCC_Roof': 'DuSA U-Net',
    'Tiled_Roof': 'DuSA U-Net',
    'Tin_Roof': 'DuSA U-Net',
    'Thatched_Roof': 'DuSA U-Net',
    'Road': 'DuSA U-Net',
    'Waterbody': 'DuSA U-Net',
    'Transformer': 'Faster R-CNN',
    'Tank': 'Faster R-CNN',
    'Well': 'DuSA U-Net'
}

# ============================================
# 3. ANALYZE EACH GEOPACKAGE
# ============================================
village_stats = []
total_by_class = {name: 0 for name in CLASS_NAMES.values()}

print("\n📋 Analyzing individual villages:")
print("-" * 90)

for gpkg in sorted(gpkg_files):
    try:
        gdf = gpd.read_file(gpkg)
        village = os.path.basename(gpkg).replace('_complete.gpkg', '').replace('_features.gpkg', '')
        size_mb = os.path.getsize(gpkg) / (1024 * 1024)
        
        print(f"\n🏘️ {village}")
        print(f"   File size: {size_mb:.2f} MB")
        print(f"   Total features: {len(gdf):,}")
        
        # Count by class
        village_classes = {}
        for class_id, class_name in CLASS_NAMES.items():
            if 'class_name' in gdf.columns:
                count = len(gdf[gdf['class_name'] == class_name])
            else:
                count = 0
            village_classes[class_name] = count
            total_by_class[class_name] += count
        
        # Display breakdown
        print(f"   Feature breakdown:")
        for name, count in village_classes.items():
            if count > 0:
                print(f"      {name}: {count:,}")
        
        village_stats.append({
            'Village': village,
            'Total': len(gdf),
            'Size_MB': size_mb,
            **village_classes
        })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

# ============================================
# 4. MODEL PERFORMANCE SUMMARY
# ============================================
print("\n" + "=" * 80)
print("📊 MODEL PERFORMANCE SUMMARY")
print("=" * 80)

print("\n📈 FEATURE EXTRACTION TOTALS:")
print("-" * 60)

model_totals = {'DuSA U-Net': 0, 'Faster R-CNN': 0}
total_features = sum(total_by_class.values())

for class_name, count in total_by_class.items():
    source = MODEL_SOURCES.get(class_name, 'Unknown')
    model_totals[source] = model_totals.get(source, 0) + count
    percentage = count / total_features * 100 if total_features > 0 else 0
    bar = '█' * int(percentage / 2)
    print(f"   {class_name:<20}: {count:>8,} ({percentage:>5.1f}%) {bar} - {source}")

print("-" * 60)
print(f"\n📊 MODEL CONTRIBUTION:")
for model, count in model_totals.items():
    percentage = count / total_features * 100 if total_features > 0 else 0
    bar = '█' * int(percentage / 2)
    print(f"   {model}: {count:>8,} features ({percentage:.1f}%) {bar}")

print(f"\n📊 TOTAL FEATURES EXTRACTED: {total_features:,}")

# ============================================
# 5. MODEL ACCURACY METRICS
# ============================================
print("\n" + "=" * 80)
print("🎯 MODEL ACCURACY METRICS")
print("=" * 80)

# DuSA U-Net metrics
UNET_ACCURACY = 97.88

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         DuSA U-NET MODEL                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Architecture: DuSA U-Net (Dual Self-Attention)                     ║
║  Parameters: 31.18M                                                 ║
║  Training Patches: 36,017                                           ║
║  Training Villages: 9                                               ║
║  Best Validation Accuracy: {UNET_ACCURACY:.2f}%                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Features Extracted by U-Net:                                       ║
║    - RCC_Roof: {total_by_class.get('RCC_Roof', 0):,}                                                 ║
║    - Thatched_Roof: {total_by_class.get('Thatched_Roof', 0):,}                                             ║
║    - Tin_Roof: {total_by_class.get('Tin_Roof', 0):,}                                                   ║
║    - Tiled_Roof: {total_by_class.get('Tiled_Roof', 0):,}                                                 ║
║    - Roads: {total_by_class.get('Road', 0):,}                                                       ║
║    - Water bodies: {total_by_class.get('Waterbody', 0):,}                                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# RCNN metrics
RCNN_PRECISION = 36.17
RCNN_RECALL = 91.89
RCNN_F1 = 51.91

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      FASTER R-CNN MODEL                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Architecture: Faster R-CNN with ResNet-50 FPN                       ║
║  Training Patches: ~500 (utility patches)                           ║
║  Classes: Transformer, Tank, Well                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Validation Metrics:                                                ║
║    Mean Precision: {RCNN_PRECISION:.2f}%                                        ║
║    Mean Recall: {RCNN_RECALL:.2f}%                                              ║
║    Mean F1-Score: {RCNN_F1:.2f}%                                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Features Extracted by R-CNN:                                       ║
║    - Transformers: {total_by_class.get('Transformer', 0):,}                                                 ║
║    - Tanks: {total_by_class.get('Tank', 0):,}                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ============================================
# 6. CREATE SUMMARY TABLE
# ============================================
if village_stats:
    df = pd.DataFrame(village_stats)
    df.to_csv('final_analysis_summary.csv', index=False)
    print("\n✅ Summary saved to: final_analysis_summary.csv")
    
    # Display summary table
    print("\n📊 VILLAGE SUMMARY TABLE:")
    print("-" * 100)
    display_cols = ['Village', 'Total', 'RCC_Roof', 'Thatched_Roof', 'Transformer', 'Tank']
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

# ============================================
# 7. CREATE VISUALIZATIONS
# ============================================
print("\n📊 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Features per village
ax1 = axes[0, 0]
if village_stats:
    villages = [s['Village'][:20] for s in village_stats]
    totals = [s['Total'] for s in village_stats]
    colors = plt.cm.viridis(np.linspace(0, 1, len(villages)))
    bars = ax1.bar(villages, totals, color=colors)
    ax1.set_xlabel('Village')
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Features Extracted per Village')
    ax1.set_xticklabels(villages, rotation=45, ha='right', fontsize=8)
    for bar, val in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                f'{val:,}', ha='center', va='bottom', fontsize=8)

# Plot 2: Feature type distribution
ax2 = axes[0, 1]
if total_by_class:
    classes = [k for k, v in total_by_class.items() if v > 0]
    counts = [v for v in total_by_class.values() if v > 0]
    if classes:
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Feature Distribution')

# Plot 3: Model contribution
ax3 = axes[1, 0]
if model_totals:
    models = list(model_totals.keys())
    counts = list(model_totals.values())
    colors = ['#2ecc71', '#e74c3c']
    ax3.bar(models, counts, color=colors)
    ax3.set_ylabel('Number of Features')
    ax3.set_title('Model Contribution')
    for i, (m, c) in enumerate(zip(models, counts)):
        ax3.text(i, c + 500, f'{c:,}', ha='center')

# Plot 4: Top features
ax4 = axes[1, 1]
if total_by_class:
    top_features = sorted([(k, v) for k, v in total_by_class.items() if v > 0], 
                         key=lambda x: x[1], reverse=True)[:8]
    if top_features:
        names, counts = zip(*top_features)
        ax4.barh(names, counts, color='skyblue')
        ax4.set_xlabel('Count')
        ax4.set_title('Top 8 Feature Types')
        for i, (n, c) in enumerate(zip(names, counts)):
            ax4.text(c + 100, i, f'{c:,}', va='center')

plt.tight_layout()
plt.savefig('final_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualization saved to: final_analysis.png")

# ============================================
# 8. FINAL REPORT
# ============================================
print("\n" + "=" * 80)
print("📝 FINAL REPORT SUMMARY")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    HACKATHON SUBMISSION READY                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Total GeoPackages: {len(gpkg_files)}                                              ║
║  Total Features: {total_features:,}                                                    ║
║  DuSA U-Net Accuracy: {UNET_ACCURACY:.2f}%                                            ║
║  Faster R-CNN Mean Recall: {RCNN_RECALL:.2f}%                                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Features Successfully Extracted:                                    ║
║    ✅ RCC_Roof: {total_by_class.get('RCC_Roof', 0):,}                                                ║
║    ✅ Thatched_Roof: {total_by_class.get('Thatched_Roof', 0):,}                                            ║
║    ✅ Tin_Roof: {total_by_class.get('Tin_Roof', 0):,}                                                  ║
║    ✅ Transformers: {total_by_class.get('Transformer', 0):,}                                                ║
║    ✅ Tanks: {total_by_class.get('Tank', 0):,}                                                      ║
║    ⚠️ Tiled_Roof, Roads, Water, Wells: Limited                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
