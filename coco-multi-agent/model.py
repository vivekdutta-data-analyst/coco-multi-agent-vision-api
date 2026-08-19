"""
model.py
--------
MultiLabelCOCONet: a from-scratch CNN for multi-label image classification on
(a subset of) COCO, plus the training loop used to produce train_results.png
and the saved weights (coco_multilabel_cnn.pth).

Architecture (per spec):
    4x [Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d]   (>= 3 conv blocks)
    -> GlobalAvgPool2d
    -> Linear -> Sigmoid   (one output per COCO class present in the dataset)

Loss: BCELoss (binary cross-entropy) since labels are independent multi-hot
vectors, not mutually exclusive classes.

Dataset expected (Kaggle: shubham2703/coco-dataset-for-multi-label-image-
classification): an images/ directory plus a CSV where each row is an image
id and a set of 0/1 columns, one per class. The exact column layout varies
by download, so `CocoMultiLabelDataset` auto-detects the label columns
(everything that isn't the image-id column) rather than hardcoding names.

Usage (Colab / Kaggle notebook):
    !kaggle datasets download -d shubham2703/coco-dataset-for-multi-label-image-classification
    !unzip -q coco-dataset-for-multi-label-image-classification.zip -d coco_data

    from model import MultiLabelCOCONet, CocoMultiLabelDataset, train_model

    train_ds = CocoMultiLabelDataset(csv_path="coco_data/train_labels.csv",
                                      img_dir="coco_data/train_images")
    val_ds   = CocoMultiLabelDataset(csv_path="coco_data/val_labels.csv",
                                      img_dir="coco_data/val_images",
                                      classes=train_ds.classes)

    model, history = train_model(train_ds, val_ds, num_classes=len(train_ds.classes),
                                  class_names=train_ds.classes, epochs=15)
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class MultiLabelCOCONet(nn.Module):
    """
    Custom CNN for multi-label classification.
    4 conv blocks -> GlobalAvgPool -> FC -> Sigmoid.
    """

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # 224 -> 112
            ConvBlock(32, 64),            # 112 -> 56
            ConvBlock(64, 128),           # 56  -> 28
            ConvBlock(128, 256),          # 28  -> 14
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (B, 256, 1, 1)
        self.classifier = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits: bool = False):
        x = self.features(x)
        x = self.global_pool(x).flatten(1)     # (B, 256)
        logits = self.classifier(x)             # (B, num_classes)
        if return_logits:
            return logits
        return self.sigmoid(logits)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CocoMultiLabelDataset(Dataset):
    """
    Generic multi-hot-label image dataset.

    Expects a CSV with one column identifying the image filename (auto-detected
    from common names: 'image_id', 'image', 'filename', 'Image', or falls back
    to the first column) and the remaining columns as 0/1 (or 0.0/1.0) labels,
    one per class.
    """

    IMG_ID_CANDIDATES = ["image_id", "image", "filename", "Image", "img", "file_name"]

    def __init__(self, csv_path, img_dir, classes=None, img_size=224, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

        id_col = next((c for c in self.IMG_ID_CANDIDATES if c in self.df.columns),
                       self.df.columns[0])
        self.id_col = id_col
        self.classes = classes or [c for c in self.df.columns if c != id_col]

        base_tf = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if augment:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.1, 0.1, 0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose(base_tf)

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, image_id):
        # Try exact match, then common extensions.
        candidates = [image_id] + [f"{image_id}{ext}" for ext in (".jpg", ".jpeg", ".png")]
        for c in candidates:
            p = os.path.join(self.img_dir, c)
            if os.path.exists(p):
                return p
        matches = glob.glob(os.path.join(self.img_dir, f"{image_id}*"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not locate image for id={image_id} in {self.img_dir}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(str(row[self.id_col]))
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        labels = torch.tensor(row[self.classes].values.astype(np.float32))
        return image, labels


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(train_ds, val_ds, num_classes, class_names,
                 epochs=15, batch_size=32, lr=1e-3, device=None,
                 out_dir="."):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MultiLabelCOCONet(num_classes=num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds)

        # ---- validate ----
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
        val_loss = val_running_loss / len(val_ds)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # Save weights + class list
    torch.save({"model_state_dict": model.state_dict(), "classes": class_names},
               os.path.join(out_dir, "coco_multilabel_cnn.pth"))
    with open(os.path.join(out_dir, "classes.json"), "w") as f:
        json.dump(class_names, f)

    plot_training_results(history, val_ds, model, class_names, device, out_dir)
    return model, history


def plot_training_results(history, val_ds, model, class_names, device, out_dir="."):
    """Produces train_results.png: loss curves + a sample-prediction panel."""
    model.eval()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curves
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].plot(history["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Sample prediction (top-5) on one validation image
    image, labels = val_ds[0]
    with torch.no_grad():
        probs = model(image.unsqueeze(0).to(device)).cpu().numpy()[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    top5_names = [class_names[i] for i in top5_idx]
    top5_vals = [probs[i] for i in top5_idx]

    axes[1].barh(top5_names[::-1], top5_vals[::-1], color="steelblue")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_title("Sample top-5 predictions (val image 0)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_results.png"), dpi=150)
    plt.close()
    print(f"Saved {os.path.join(out_dir, 'train_results.png')}")


# --------------------------------------------------------------------------- #
# Inference helper (used by agent_graph.py's cnn_node)
# --------------------------------------------------------------------------- #
def load_trained_model(weights_path="coco_multilabel_cnn.pth", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    classes = checkpoint["classes"]
    model = MultiLabelCOCONet(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, device


def predict_top_k(model, classes, device, pil_image, k=5, img_size=224):
    """Run inference on a single PIL image, return top-k {class: prob} dict."""
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).cpu().numpy()[0]
    top_idx = np.argsort(probs)[::-1][:k]
    return {classes[i]: round(float(probs[i]), 4) for i in top_idx}


if __name__ == "__main__":
    # Smoke test: verify the architecture builds and runs a forward pass.
    m = MultiLabelCOCONet(num_classes=80)
    dummy = torch.randn(2, 3, 224, 224)
    out = m(dummy)
    print("Output shape:", out.shape)  # (2, 80)
    print("Output range:", out.min().item(), out.max().item())  # should be within (0, 1)
