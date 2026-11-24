"""
Training script for Stage 2: Shared-weight Siamese ResNet18 Damage Classifier.

Architecture:
  - Shared-weight ImageNet-pretrained ResNet18 encoder.
  - Extracts 512-d feature vectors: f_pre and f_post.
  - Concatenates [f_pre, f_post, f_post - f_pre] (1536-d).
  - Head: Dropout(0.3) + Linear(1536, 4) predicting the 4 damage tiers.

Loss: CrossEntropyLoss with label_smoothing=0.05
Optimizer: AdamW + CosineAnnealingLR + Mixed Precision (AMP)
Evaluation: Macro F1 across the 4 damage classes + validation confusion matrix.
Best Checkpoint: Saved to artifacts/stage2_best.pth based on macro F1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from src.stage2_dataset import (
    DAMAGE_LEVELS,
    DAMAGE_TO_IDX,
    IDX_TO_DAMAGE,
    Stage2SiameseCropDataset,
    get_stage2_dataloaders,
)


class SiameseDamageClassifier(nn.Module):
    """Shared-weight Siamese ResNet18 damage tier classifier."""

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Truncate fc and avgpool: keep layers up to layer4 + adaptive avgpool
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_dim = 512

        # Fuse [f_pre, f_post, f_post - f_pre] -> 512 * 3 = 1536
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim * 3, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-d feature embedding."""
        feat = self.encoder(x)
        return torch.flatten(feat, 1)

    def forward(self, pre_crop: torch.Tensor, post_crop: torch.Tensor) -> torch.Tensor:
        """Compute Siamese embeddings, difference fusion, and damage logits."""
        f_pre = self.extract_features(pre_crop)
        f_post = self.extract_features(post_crop)
        diff = f_post - f_pre
        fused = torch.cat([f_pre, f_post, diff], dim=1)  # [B, 1536]
        logits = self.classifier(fused)  # [B, 4]
        return logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 15,
    use_amp: bool = True,
) -> float:
    """Run one training epoch over paired crops with progress bar."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Train Ep [{epoch:02d}/{total_epochs:02d}]", leave=False)

    for pre, post, labels, _ in pbar:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(pre, post)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pre, post)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, str]:
    """Evaluate loss, macro F1, and confusion matrix on the validation set with progress bar."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    pbar = tqdm(loader, desc="Evaluating Val", leave=False)

    for pre, post, labels, _ in pbar:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(pre, post)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

    val_loss = total_loss / max(len(loader), 1)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(DAMAGE_LEVELS))))
    report_str = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(DAMAGE_LEVELS))),
        target_names=DAMAGE_LEVELS,
        digits=3,
        zero_division=0,
    )

    return val_loss, macro_f1, conf_mat, report_str


def print_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """Pretty-print confusion matrix with formatted rows."""
    col_headers = "".join(f"{name[:8]:>10s}" for name in class_names)
    header = f"{'True / Pred':16s}" + col_headers
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name[:14]:16s}" + "".join(f"{cm[i, j]:10d}" for j in range(len(class_names)))
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 Siamese Damage Classifier")
    parser.add_argument("--data_root", type=str, default="Dataset/CIT_Hackathon_PS04_aerial/train")
    parser.add_argument("--splits_file", type=str, default="artifacts/splits.json")
    parser.add_argument("--save_path", type=str, default="artifacts/stage2_best.pth")
    parser.add_argument("--history_path", type=str, default="artifacts/stage2_history.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.10)
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--pad_factor", type=float, default=1.3)
    parser.add_argument("--num_samples", type=int, default=12000, help="Samples per epoch for balanced training")
    parser.add_argument("--sampler_power", type=float, default=0.5, help="0.5=inverse sqrt, 1.0=inverse frequency")
    default_workers = 0 if os.name == "nt" else 4
    parser.add_argument("--num_workers", type=int, default=default_workers)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_weighted_sampler", action="store_true", help="Disable WeightedRandomSampler")
    args = parser.parse_args()

    print("=" * 64)
    print("  PS04 Stage 2: Siamese ResNet18 Damage Classifier")
    print("=" * 64)
    print(f"Device:           {args.device}")
    print(f"Data root:        {args.data_root}")
    print(f"Splits file:      {args.splits_file}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"Label smoothing:  {args.label_smoothing}")
    print(f"Crop size:        {args.crop_size}x{args.crop_size} (pad factor {args.pad_factor}x)")
    print(f"Checkpoint save:  {args.save_path}")
    print("=" * 64)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load Data
    train_ds, val_ds, train_loader, val_loader = get_stage2_dataloaders(
        splits_path=args.splits_file,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        pad_factor=args.pad_factor,
        use_weighted_sampler=not args.no_weighted_sampler,
        sampler_power=args.sampler_power,
        num_samples=args.num_samples,
    )

    print(f"Loaded {len(train_ds)} train crops, {len(val_ds)} val crops.")
    print("Train Class Distribution:", train_ds.get_class_counts())
    print("Val Class Distribution:  ", val_ds.get_class_counts())

    # 2. Build Model
    model = SiameseDamageClassifier(
        pretrained=True,
        num_classes=4,
        dropout=args.dropout,
    ).to(device)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Training Loop & History Tracking
    best_macro_f1 = 0.0
    start_time = time.time()
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
    }
    history_path = Path(args.history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            use_amp=(device.type == "cuda"),
        )
        scheduler.step()

        val_loss, val_macro_f1, conf_mat, report_str = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        ep_duration = time.time() - ep_start
        improved = val_macro_f1 > best_macro_f1

        history["epoch"].append(epoch)
        history["train_loss"].append(round(float(train_loss), 4))
        history["val_loss"].append(round(float(val_loss), 4))
        history["val_macro_f1"].append(round(float(val_macro_f1), 4))

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if improved:
            best_macro_f1 = val_macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_macro_f1,
                    "val_loss": val_loss,
                    "confusion_matrix": conf_mat.tolist(),
                    "crop_size": args.crop_size,
                    "pad_factor": args.pad_factor,
                },
                save_path,
            )
            marker = " ★ BEST"
        else:
            marker = ""

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f} | "
            f"Time: {ep_duration:.1f}s{marker}"
        )

        # Print detailed report on improved epochs or final epoch
        if improved or epoch == args.epochs:
            print("\nClassification Report (Val):")
            print(report_str)
            print("Confusion Matrix:")
            print_confusion_matrix(conf_mat, DAMAGE_LEVELS)
            print("-" * 64)

    total_time = time.time() - start_time
    print("=" * 64)
    print(f"Training completed in {total_time / 60:.1f} min.")
    print(f"Best Validation Macro F1: {best_macro_f1:.4f}")
    print(f"Model saved to: {save_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
