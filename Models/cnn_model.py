import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import os
import pickle
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "ncr_satellite_data"
LABELS_CSV = PROJECT_ROOT / "data" / "rain_cnn_labels.csv"

MODEL_PATH = PROJECT_ROOT / "Rain_Models" / "rain_cnn.pt"
HISTORY_PKL = PROJECT_ROOT / "Rain_Models" / "rain_cnn_training_history.pkl"

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
print(f"Image directory: {IMAGE_DIR}")
print(f"Labels file: {LABELS_CSV}")

# ---- CONFIG ----
from pathlib import Path

# Project root = Actual_APP
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "ncr_satellite_data"
LABELS_CSV = PROJECT_ROOT / "data" / "rain_cnn_labels.csv"

MODEL_PATH = PROJECT_ROOT / "Rain_Models" / "rain_cnn.pt"
HISTORY_PKL = PROJECT_ROOT / "Rain_Models" / "rain_cnn_training_history.pkl"
BATCH_SIZE = 8
EPOCHS = 15
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---- DATASET ----
class RainDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row["filename"])).convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = self.transform(bg)
        label = torch.tensor(row["rained"], dtype=torch.float32)
        return img, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- LOAD + STRATIFIED SPLIT ----
df = pd.read_csv(LABELS_CSV)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["rained"], random_state=42
)

print(f"Train: {len(train_df)} images, rain distribution:\n{train_df['rained'].value_counts()}")
print(f"Val: {len(val_df)} images, rain distribution:\n{val_df['rained'].value_counts()}")

train_ds = RainDataset(train_df, IMAGE_DIR, train_transform)
val_ds = RainDataset(val_df, IMAGE_DIR, val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---- MODEL: transfer learning on ResNet18 ----
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

# ---- WEIGHTED LOSS for class imbalance ----
no_rain_count = (df["rained"] == 0).sum()
rain_count = (df["rained"] == 1).sum()
pos_weight = torch.tensor([no_rain_count / rain_count], dtype=torch.float32).to(device)
print(f"pos_weight (no_rain/rain): {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# ---- TRAIN LOOP ----
history = {
    "train_loss": [],
    "val_acc": [],
    "tp": [], "tn": [], "fp": [], "fn": [],
}

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ---- VALIDATION with per-class breakdown ----
    model.eval()
    correct, total = 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    avg_loss = train_loss / len(train_loader)
    acc = correct / total if total > 0 else 0

    history["train_loss"].append(avg_loss)
    history["val_acc"].append(acc)
    history["tp"].append(tp)
    history["tn"].append(tn)
    history["fp"].append(fp)
    history["fn"].append(fn)

    print(f"Epoch {epoch+1}/{EPOCHS} — train loss: {avg_loss:.4f}, "
          f"val acc: {acc:.2%}  |  TP:{tp} TN:{tn} FP:{fp} FN:{fn}")

# ---- SAVE MODEL WEIGHTS ----
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel weights saved to {MODEL_PATH}")

# ---- SAVE TRAINING HISTORY + METADATA AS PKL ----
pkl_data = {
    "history": history,
    "final_val_acc": history["val_acc"][-1],
    "pos_weight": pos_weight.item(),
    "train_size": len(train_df),
    "val_size": len(val_df),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "class_distribution": {
        "no_rain": int(no_rain_count),
        "rain": int(rain_count),
    },
}

with open(HISTORY_PKL, "wb") as f:
    pickle.dump(pkl_data, f)

print(f"Training history + metadata saved to {HISTORY_PKL}")