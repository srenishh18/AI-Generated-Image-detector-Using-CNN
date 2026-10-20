import os
import sys
import json
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def log(msg):
    print(msg, flush=True)

# ---------------------------------------------------------
# 1. Dataset Loading & Preprocessing Class
# ---------------------------------------------------------
class AIvsRealDataset(Dataset):
    def __init__(self, root_dir, img_size=128, transform=None):
        self.img_size = img_size
        self.transform = transform
        self.image_paths = []
        self.labels = []  # 0: Real Image, 1: AI-Generated Image

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        # Load AI images (label = 1)
        ai_dir = os.path.join(root_dir, 'AI')
        for root, _, files in os.walk(ai_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    self.image_paths.append(os.path.join(root, f))
                    self.labels.append(1)

        # Load Real images (label = 0)
        real_dir = os.path.join(root_dir, 'Real')
        for root, _, files in os.walk(real_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    self.image_paths.append(os.path.join(root, f))
                    self.labels.append(0)

        log(f"[*] Dataset Loaded: Total={len(self.labels)} | AI={sum(self.labels)} | Real={len(self.labels)-sum(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                # PREPROCESSING / SHORTCUT PREVENTION:
                # Resize all images uniformly to (img_size, img_size) to prevent the CNN 
                # from using native image dimensions/resolutions as a classification shortcut.
                if img.size != (self.img_size, self.img_size):
                    img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                image = img.copy()
        except Exception:
            # Fallback for corrupt file handling
            image = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# ---------------------------------------------------------
# 2. Custom 5-Layer Convolutional Neural Network (CNN)
# ---------------------------------------------------------
# Architecture: Conv2D -> BatchNorm -> ReLU -> MaxPool2D -> AdaptiveAvgPool -> Linear
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Layer 1: Extract low-level edges & color textures (3 -> 32 channels)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # OVERFITTING PREVENTION: Stabilizes training activations & regularizes
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Spatial reduction (128x128 -> 64x64)

            # Layer 2: Extract mid-level patterns & shapes (32 -> 64 channels)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # OVERFITTING PREVENTION: Batch Normalization
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Spatial reduction (64x64 -> 32x32)

            # Layer 3: Extract high-level structural features (64 -> 128 channels)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # OVERFITTING PREVENTION: Batch Normalization
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Spatial reduction (32x32 -> 16x16)

            # Layer 4: Extract complex AI generator artifacts (128 -> 256 channels)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # OVERFITTING PREVENTION: Batch Normalization
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Spatial reduction (16x16 -> 8x8)

            # Layer 5: Deep feature aggregation (256 -> 512 channels)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # OVERFITTING PREVENTION: Batch Normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Pooling: reduces spatial dimension to 1x1 vector
        )
        self.classifier = nn.Sequential(
            # OVERFITTING PREVENTION: Dropout randomly zeroes 30% of features during training
            # to prevent neuron co-adaptation and force generalized learning.
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Final binary output logit (positive = AI, negative = Real)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(1)

class EfficientNetB0Detector(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB0Detector, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),  # OVERFITTING PREVENTION: Dropout
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x.squeeze(1)

# ---------------------------------------------------------
# 3. Model Training & Evaluation Engine
# ---------------------------------------------------------
def train_and_evaluate(model_type='custom', epochs=3, batch_size=64, lr=1e-3, img_size=128):
    dataset_dir = r"d:\AI Image detector\dataset"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[*] Compute Device: {device}")

    # OVERFITTING PREVENTION & DATA AUGMENTATION:
    # Randomly flips, rotates, and adjusts brightness/contrast during training.
    # Forces the network to learn invariant features rather than memorizing exact pixel arrangements.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),      # Data Augmentation
        transforms.RandomRotation(degrees=10),       # Data Augmentation
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = AIvsRealDataset(dataset_dir, img_size=img_size, transform=None)
    total_size = len(full_dataset)
    
    # DATASET SPLITTING (80% Train, 10% Validation, 10% Unseen Test):
    train_size = int(0.80 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducible splits
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_test_transform
    test_subset.dataset.transform = val_test_transform

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    log(f"[*] Splits: Train={len(train_subset)} | Val={len(val_subset)} | Test={len(test_subset)}")

    if model_type == 'efficientnet':
        log("[*] Model Architecture: EfficientNet-B0 CNN")
        model = EfficientNetB0Detector(pretrained=True).to(device)
    else:
        log("[*] Model Architecture: Custom 5-Layer CNN (Conv2D -> BatchNorm -> ReLU -> MaxPool2D -> AdaptiveAvgPool -> Linear)")
        model = CustomCNN().to(device)

    # CLASS BALANCE HANDLING:
    # 5422 Real images vs 3506 AI images (~60/40 ratio).
    # pos_weight adjusts Binary Cross-Entropy Loss to penalize AI errors proportionally,
    # preventing majority class bias.
    pos_weight = torch.tensor([5422.0 / 3506.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # OVERFITTING PREVENTION: Weight Decay (L2 Regularization)
    # Penalizes large weight values in the AdamW optimizer to maintain simple, generalizable decisions.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    log("\n" + "="*60)
    log("  STARTING MODEL TRAINING & METRICS EVALUATION")
    log("="*60)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        log(f"\n>>> EPOCH [{epoch+1}/{epochs}] <<<")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = (preds == labels).sum().item() / labels.size(0)
                log(f"  Step [{batch_idx+1:03d}/{len(train_loader):03d}] | Loss: {loss.item():.4f} | Batch Accuracy: {batch_acc*100:.1f}%")

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        scheduler.step(epoch_val_loss)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        log(f"  --> Epoch {epoch+1} Results:")
        log(f"      Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        log(f"      Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc*100:.2f}%")

        # CHECKPOINT SAVING: Saves weights only when Validation Loss improves
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'img_size': img_size
            }, r"d:\AI Image detector\best_model.pth")
            log(f"      [+] Checkpoint Saved! (Val Loss: {epoch_val_loss:.4f})")

    duration = time.time() - start_time
    log(f"\n[*] Total Training Time: {duration:.2f} seconds.")

    # ---------------------------------------------------------
    # 4. Final Evaluation on Unseen Test Dataset (10%)
    # ---------------------------------------------------------
    log("\n" + "="*60)
    log("  TERMINAL EVALUATION REPORT (UNSEEN TEST SET)")
    log("="*60)
    checkpoint = torch.load(r"d:\AI Image detector\best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds_probs = []
    test_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            test_preds_probs.extend(probs)
            test_true_labels.extend(labels.numpy())

    test_preds_probs = np.array(test_preds_probs)
    test_true_labels = np.array(test_true_labels)
    test_preds_binary = (test_preds_probs >= 0.5).astype(int)

    acc = accuracy_score(test_true_labels, test_preds_binary)
    prec = precision_score(test_true_labels, test_preds_binary, zero_division=0)
    rec = recall_score(test_true_labels, test_preds_binary, zero_division=0)
    f1 = f1_score(test_true_labels, test_preds_binary, zero_division=0)
    roc_auc = roc_auc_score(test_true_labels, test_preds_probs)
    cm = confusion_matrix(test_true_labels, test_preds_binary)

    metrics_dict = {
        'model_type': model_type,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'best_val_loss': float(best_val_loss),
        'test_count': int(len(test_true_labels)),
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        },
        'history': history
    }

    log(f"  Test Accuracy:    {acc * 100:.2f}%")
    log(f"  Precision:        {prec * 100:.2f}%")
    log(f"  Recall:           {rec * 100:.2f}%")
    log(f"  F1-Score:         {f1 * 100:.2f}%")
    log(f"  ROC-AUC Score:    {roc_auc:.4f}")
    log(f"  Best Val Loss:    {best_val_loss:.4f}")
    log("\n  CONFUSION MATRIX TABLE:")
    log(f"                    Predicted Real   Predicted AI")
    log(f"    Actual Real       {cm[0,0]:<14} {cm[0,1]:<12}")
    log(f"    Actual AI         {cm[1,0]:<14} {cm[1,1]:<12}")

    with open(r"d:\AI Image detector\metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    log("\n[+] Saved best model weights to d:\\AI Image detector\\best_model.pth")
    log("[+] Saved metrics summary to d:\\AI Image detector\\metrics.json")

if __name__ == '__main__':
    model_choice = sys.argv[1] if len(sys.argv) > 1 else 'custom'
    epoch_count = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    img_sz = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    train_and_evaluate(model_type=model_choice, epochs=epoch_count, batch_size=64, img_size=img_sz)
