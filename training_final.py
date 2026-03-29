
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import glob
import sys
sys.path.append('/nfsshare/users/manjula/svamitva_project')
from models.dusa_unet import DuSA_UNet

# Force use GPU 5
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("RESUMING OPTIMIZED TRAINING")
print("=" * 60)

# ============================================
# DATA AUGMENTATION
# ============================================
class PatchDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=True):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        self.augment = augment
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
            A.ElasticTransform(p=0.2, alpha=1, sigma=50),
            A.CLAHE(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        print(f"Loaded {len(self.images)} patches")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert('RGB'))
        mask = np.array(Image.open(self.masks[idx]))
        mask = np.clip(mask, 0, 9)
        
        if self.augment:
            augmented = self.transform(image=img, mask=mask)
        else:
            augmented = self.val_transform(image=img, mask=mask)
            
        return augmented['image'], augmented['mask'].long()

# ============================================
# LABEL SMOOTHING LOSS
# ============================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = nn.functional.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))

# ============================================
# LOAD DATA
# ============================================
print("\nLOADING DATA")
dataset = PatchDataset("data/training/patches/images", "data/training/patches/masks", augment=True)

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Disable augmentation for validation
val_dataset.dataset.augment = False

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

print(f"\nTrain: {len(train_dataset)} patches")
print(f"Val: {len(val_dataset)} patches")
print(f"Batch size: {batch_size}")

# ============================================
# MODEL
# ============================================
print("\nCREATING MODEL")
device = torch.device('cuda')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = DuSA_UNet(n_classes=10).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ============================================
# RESUME FROM SAVED MODEL
# ============================================
start_epoch = 0
best_acc = 0

# Try to load the best saved model
model_paths = [
    'outputs/best_model_optimized.pth',
    'outputs/best_model_final.pth', 
    'outputs/best_model.pth'
]

for path in model_paths:
    if os.path.exists(path):
        print(f"\n📂 Loading model from: {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded model successfully")
        best_acc = 91.75  # Your last known best accuracy
        start_epoch = 12  # Resume from epoch 13
        break

if start_epoch == 0:
    print("\n🆕 No saved model found. Starting from scratch")

# ============================================
# LOSS & OPTIMIZER
# ============================================
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# OneCycleLR scheduler
total_steps = len(train_loader) * 50
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=total_steps, 
    pct_start=0.3, anneal_strategy='cos'
)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

os.makedirs("outputs", exist_ok=True)

print("\n" + "=" * 60)
print(f"RESUMING TRAINING FROM EPOCH {start_epoch + 1}/50")
print(f"Current best accuracy: {best_acc:.2f}%")
print("Optimizations: Label Smoothing, OneCycleLR, Mixed Precision")
print("=" * 60)

# ============================================
# TRAINING LOOP
# ============================================
for epoch in range(start_epoch, 50):
    model.train()
    train_loss = 0
    
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            out = model(img)
            loss = criterion(out, mask)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            
            with torch.cuda.amp.autocast():
                out = model(img)
                loss = criterion(out, mask)
            
            val_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            correct += (pred == mask).sum().item()
            total += mask.numel()
    
    val_loss /= len(val_loader)
    accuracy = correct / total * 100
    scheduler.step()
    
    print(f"Epoch {epoch+1:3d}/50 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {accuracy:.2f}%")
    
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "outputs/best_model_resumed.pth")
        print(f"  ✅ New best! Accuracy: {accuracy:.2f}%")
    
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print(f"🏆 BEST VALIDATION ACCURACY: {best_acc:.2f}%")
print("💾 Model saved: outputs/best_model_resumed.pth")
print("=" * 60)
