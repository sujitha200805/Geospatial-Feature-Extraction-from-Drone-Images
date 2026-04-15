import os
import cv2
import numpy as np
from tqdm import tqdm

image_dir = 'data/training/patches/images'
mask_dir = 'data/training/patches/masks'

imgs = sorted(os.listdir(image_dir))
tank_files = []

print("Scanning for the 186 tank patches...")
for f in tqdm(imgs, desc="Scanning"):
    m_path = os.path.join(mask_dir, f)
    if not os.path.exists(m_path): continue
    mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None and 8 in mask:
        tank_files.append((f, mask))

print(f"Found {len(tank_files)} tank patches. Generating augments...")

aug_count = 0
target_augs_per_img = 5 # 186 * 5 = 930 augmented images

for f, mask in tqdm(tank_files, desc="Augmenting Tanks"):
    i_path = os.path.join(image_dir, f)
    img = cv2.imread(i_path)
    
    for _ in range(target_augs_per_img):
        flip_code = np.random.choice([-2, -1, 0, 1]) 
        if flip_code != -2:
            aug_img = cv2.flip(img, int(flip_code))
            aug_mask = cv2.flip(mask, int(flip_code))
        else:
            aug_img = img.copy()
            aug_mask = mask.copy()
            
        alpha = np.random.uniform(0.7, 1.4) 
        beta = np.random.uniform(-40, 40)   
        aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
        
        new_f = f"aug_tank_{aug_count}_{f}"
        cv2.imwrite(os.path.join(image_dir, new_f), aug_img)
        cv2.imwrite(os.path.join(mask_dir, new_f), aug_mask)
        aug_count += 1

print(f"Successfully generated {aug_count} highly-augmented tank patches!")
