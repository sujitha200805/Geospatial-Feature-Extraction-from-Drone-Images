import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T

class UtilityDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.image_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        
        # we only keep masks that actually have utilities in them to speed up training,
        # or we could keep all. Let's list all files.
        self.imgs = list(sorted(os.listdir(self.image_dir)))
        
        # Mappings: Background=0, Transformer=1 (mask 7), Tank=2 (mask 8), Well=3 (mask 9)
        self.mask_to_class = {7: 1, 8: 2, 9: 3}

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.imgs[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        boxes = []
        labels = []
        
        # Extract bounding boxes for each utility class
        for mask_val, class_idx in self.mask_to_class.items():
            # Create binary mask for this class
            binary_mask = (mask_img == mask_val).astype(np.uint8)
            
            # Morphological dilation to inflate 1-pixel utilities into larger bounding boxes
            # Increased from 5x5 to 17x17 so Faster R-CNN's default 32x32 anchors can detect them
            kernel = np.ones((17, 17), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            
            try:
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            except cv2.error:
                # In some very rare cases of invalid mask shapes, connected componets fail
                continue
            
            # stats[0] is the background component
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Faster R-CNN takes boxes as [xmin, ymin, xmax, ymax]
                xmin = float(x)
                ymin = float(y)
                xmax = float(x + w)
                ymax = float(y + h)
                
                # Check box bounds to make sure the boxes aren't collapsing into invalid shapes
                if xmax > xmin and ymax > ymin and w >= 2 and h >= 2:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_idx)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        
        target = {}
        if len(boxes) > 0:
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Handle empty images (no utilities)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["image_id"] = image_id
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms is not None:
            # Note: torchvision target transforms are complex. Here we manually convert img to tensor.
            pass
            
        img = T.ToTensor()(img) # Converts [H,W,C] to [C,H,W]

        return img, target

    def __len__(self):
        return len(self.imgs)

def compute_dist(box1, box2):
    import math
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return math.hypot(c1_x - c2_x, c1_y - c2_y)

def get_model(num_classes):
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    
    # Anchor sizes tailored for 16x16 to 256x256 pixel objects (FPN needs 5 feature map sizes)
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # load a model pre-trained on COCO
    # We apply NMS of 0.2 to harshly suppress duplicating boxes on the same object
    model = fasterrcnn_resnet50_fpn(
        weights='DEFAULT', 
        rpn_anchor_generator=anchor_generator,
        box_nms_thresh=0.2
    )
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    print("Initializing Faster R-CNN on Utility Patches...")
    
    # Num classes = 4 (Background + Transformer + Tank + Well)
    num_classes = 4
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize Dataset
    dataset = UtilityDataset(root='data/training/patches')
    
    # Subset simply if we just want a small test or whole set 
    # Let's filter out images without utilities to speed up training since most patches are empty
    print(f"Scanning {len(dataset)} total patches for utilities to build active subset...")
    utilities_indices = []
    empty_indices = []
    
    # It might take a moment to scan. Let's do a quick random subset or read masks
    # Alternatively we can scan directory quickly
    mask_dir = 'data/training/patches/masks'
    valid_files = sorted(os.listdir(mask_dir))
    
    import tqdm
    for i, file in enumerate(tqdm.tqdm(valid_files, desc="Scanning masks", unit="file")):
        mask_path = os.path.join(mask_dir, file)
        # We know classes 7, 8, 9 are utilities.
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None: continue
        # Check if any utility pixel exists that forms a valid bounding box (w>=2, h>=2)
        has_valid_utility = False
        if np.any(np.isin(mask_img, [7, 8, 9])):
            for mask_val in [7, 8, 9]:
                binary_mask = (mask_img == mask_val).astype(np.uint8)
                
                # Morphological dilation to inflate 1-pixel utilities into larger bounding boxes
                # Increased from 5x5 to 17x17 so Faster R-CNN's default 32x32 anchors can detect them
                kernel = np.ones((17, 17), np.uint8)
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
                
                try:
                    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                    for j in range(1, num_labels):
                        w = stats[j, cv2.CC_STAT_WIDTH]
                        h = stats[j, cv2.CC_STAT_HEIGHT]
                        if w >= 2 and h >= 2:
                            has_valid_utility = True
                            break
                except cv2.error:
                    pass
                if has_valid_utility: break
        
        if has_valid_utility:
            utilities_indices.append(i)
        else:
            empty_indices.append(i)
            
    print(f"Found {len(utilities_indices)} patches with actual utilities.")
    
    # Inject hard negative empty patches (~20% of dataset proportion)
    num_empty = int(len(utilities_indices) * 0.20)
    import random
    random.seed(42)
    random.shuffle(empty_indices)
    active_indices = utilities_indices + empty_indices[:num_empty]
    print(f"Added {num_empty} empty background patches for hard-negative penalization.")
    
    if len(utilities_indices) == 0:
        print("No utilities found in patches. Exiting.")
        return
        
    dataset_subset = torch.utils.data.Subset(dataset, active_indices)
    
    # Split train/val
    train_size = int(0.8 * len(dataset_subset))
    val_size = len(dataset_subset) - train_size
    generator = torch.Generator().manual_seed(42) # using fixed seed to recreate split exactly
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_subset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn, pin_memory=False
    )
    
    torch.cuda.empty_cache()
    model = get_model(num_classes).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    
    num_epochs = 40
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass provides dict of losses for Faster R-CNN in training mode
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            
            # Clip the gradients to prevent them from skyrocketing into infinity (NaN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            epoch_loss += losses.item()
            
        lr_scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
        
        # --- Validation Block at end of Epoch ---
        model.eval()
        class_correct = {1: 0, 2: 0, 3: 0}
        class_total = {1: 0, 2: 0, 3: 0}
        class_pred_total = {1: 0, 2: 0, 3: 0}
        
        with torch.no_grad():
            for v_images, v_targets in val_loader:
                v_images = list(img.to(device) for img in v_images)
                v_targets = [{k: v.to(device) for k, v in t.items()} for t in v_targets]
                
                outputs = model(v_images)
                
                for i, output in enumerate(outputs):
                    gt_boxes = v_targets[i]['boxes'].cpu().numpy()
                    gt_labels = v_targets[i]['labels'].cpu().numpy()
                    
                    pred_boxes = output['boxes'].cpu().numpy()
                    pred_labels = output['labels'].cpu().numpy()
                    pred_scores = output['scores'].cpu().numpy()
                    
                    # Filter: using real production threshold 0.50
                    mask = pred_scores >= 0.50
                    pred_boxes = pred_boxes[mask]
                    pred_labels = pred_labels[mask]
                    
                    for cls in [1, 2, 3]:
                        gt_cls_mask = gt_labels == cls
                        pred_cls_mask = pred_labels == cls
                        
                        gt_cls_boxes = gt_boxes[gt_cls_mask]
                        pred_cls_boxes = pred_boxes[pred_cls_mask]
                        
                        class_total[cls] += len(gt_cls_boxes)
                        class_pred_total[cls] += len(pred_cls_boxes)
                        
                        matched_gt = set()
                        for p_box in pred_cls_boxes:
                            best_dist = 999999
                            best_gt_idx = -1
                            for g_idx, g_box in enumerate(gt_cls_boxes):
                                if g_idx in matched_gt: continue
                                dist = compute_dist(p_box, g_box)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_gt_idx = g_idx
                            
                            # Log correct match (center within 10 pixels of ground truth)
                            if best_dist <= 10.0:
                                class_correct[cls] += 1
                                matched_gt.add(best_gt_idx)

        total_correct = sum(class_correct.values())
        total_gt = sum(class_total.values())
        total_predicted = sum(class_pred_total.values())
        
        m_precision = total_correct / total_predicted if total_predicted > 0 else 0
        m_recall = total_correct / total_gt if total_gt > 0 else 0
        
        print(f" -> Validation:  Precision: {m_precision:.2%} | Recall: {m_recall:.2%} | GT Utilities: {total_gt} | Predicted: {total_predicted}")
        # -------------------------------------
        
    os.makedirs('outputs/final', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/final/faster_rcnn_utilities.pth')
    print("Training Complete. Model saved to outputs/final/faster_rcnn_utilities.pth")

if __name__ == '__main__':
    main()
