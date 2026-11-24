"""
Training script for Stage 1: Pre-disaster Building Instance Localization.

Model Architecture:
  segmentation_models_pytorch.Unet(
      encoder_name="resnet34",
      encoder_weights="imagenet",
      in_channels=3,
      classes=1
  )

Loss: BCEWithLogitsLoss + DiceLoss
Optimization: AdamW + CosineAnnealingLR + Mixed Precision (AMP)
Best Model: Saved to artifacts/stage1_best.pth based on validation Dice score.

Can be run locally on CPU/GPU or uploaded directly to Kaggle / Google Colab T4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

from src.stage1_dataset import Stage1BuildingDataset, get_stage1_dataloaders


class DiceLoss(nn.Module):
    """Dice Loss for binary mask segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


def compute_dice_score(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7
) -> float:
    """Compute binary Dice coefficient on predicted logits."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum().item()
    denom = preds_flat.sum().item() + targets_flat.sum().item()
    if denom == 0:
        return 1.0
    return (2.0 * intersection + eps) / (denom + eps)


def build_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Instantiate the SMP Unet model."""
    if smp is None:
        raise ImportError(
            "segmentation_models_pytorch is required. Install via `pip install segmentation-models-pytorch`."
        )
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_fn: nn.Module,
    dice_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 20,
    use_amp: bool = True,
) -> float:
    """Run one training epoch with mixed precision and progress bar."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Train Ep [{epoch:02d}/{total_epochs:02d}]", leave=False)

    for imgs, masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss_bce = bce_fn(logits, masks)
                loss_dice = dice_fn(logits, masks)
                loss = loss_bce + loss_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss_bce = bce_fn(logits, masks)
            loss_dice = dice_fn(logits, masks)
            loss = loss_bce + loss_dice
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    bce_fn: nn.Module,
    dice_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate validation loss and Dice coefficient on full resolution with progress bar."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    pbar = tqdm(loader, desc="Evaluating Val", leave=False)

    for imgs, masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss_bce = bce_fn(logits, masks)
        loss_dice = dice_fn(logits, masks)
        loss = loss_bce + loss_dice

        dice = compute_dice_score(logits, masks, threshold=0.5)

        total_loss += loss.item()
        total_dice += dice
        pbar.set_postfix({"dice": f"{dice:.4f}"})

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 Building Localization Unet")
    parser.add_argument("--data_root", type=str, default="Dataset/CIT_Hackathon_PS04_aerial/train")
    parser.add_argument("--splits_file", type=str, default="artifacts/splits.json")
    parser.add_argument("--save_path", type=str, default="artifacts/stage1_best.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--border_px", type=int, default=2)
    default_workers = 0 if os.name == "nt" else 4
    parser.add_argument("--num_workers", type=int, default=default_workers)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    args = parser.parse_args()

    print("=" * 64)
    print("  PS04 Stage 1: Building Instance Localization (Unet-ResNet34)")
    print("=" * 64)
    print(f"Device:           {args.device}")
    print(f"Data root:        {args.data_root}")
    print(f"Splits file:      {args.splits_file}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"Native crop:      {args.crop_size}x{args.crop_size}")
    print(f"Boundary border:  {args.border_px}px")
    print(f"Checkpoint save:  {args.save_path}")
    print("=" * 64)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # 1. Load Data
    _, _, train_loader, val_loader = get_stage1_dataloaders(
        splits_path=args.splits_file,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_batch_size=1,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        border_px=args.border_px,
    )
    print(f"Loaded {len(train_loader.dataset)} train pairs, {len(val_loader.dataset)} val pairs.")

    # 2. Build Model
    model = build_model(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)

    # 3. Optimization & Loss
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Training Loop
    best_dice = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            bce_fn=bce_fn,
            dice_fn=dice_fn,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            use_amp=(device.type == "cuda"),
        )
        scheduler.step()

        val_loss, val_dice = evaluate(
            model=model,
            loader=val_loader,
            bce_fn=bce_fn,
            dice_fn=dice_fn,
            device=device,
        )

        ep_duration = time.time() - ep_start
        lr_curr = optimizer.param_groups[0]["lr"]

        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                    "encoder": args.encoder,
                    "crop_size": args.crop_size,
                    "border_px": args.border_px,
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
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {lr_curr:.2e} | "
            f"Time: {ep_duration:.1f}s{marker}"
        )

    total_time = time.time() - start_time
    print("=" * 64)
    print(f"Training completed in {total_time / 60:.1f} min.")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Model saved to: {save_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
