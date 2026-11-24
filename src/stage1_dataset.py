"""
Stage 1 Dataset: Boundary-carved mask rasterization and native-resolution 512x512 crop loader.

Key Features:
1. Rasterizes building polygons from GT labels into binary masks.
2. Carves a 2-pixel black border around each building polygon using polyline outlines,
   preventing touching rooftops from merging in the mask representation.
3. Samples native-resolution 512x512 random crops during training with flips and 90-degree rotations,
   preserving fine structural rooftop details without downscaling blur.
4. Preserves full 1024x1024 resolution during validation / evaluation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data_utils import parse_wkt_polygon, polygon_centroid

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

# Standard ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def rasterize_boundary_carved_mask(
    polygons: Sequence[Sequence[Sequence[float]]],
    height: int = 1024,
    width: int = 1024,
    border_px: int = 2,
) -> np.ndarray:
    """Rasterize building polygons into a binary mask with carved boundaries.

    Parameters
    ----------
    polygons : Sequence[Sequence[Sequence[float]]]
        List of polygon vertex coordinates [[[x1, y1], [x2, y2], ...], ...].
    height, width : int
        Dimensions of output mask (default 1024x1024).
    border_px : int
        Pixel thickness of the boundary carving (default 2).

    Returns
    -------
    np.ndarray (uint8, shape [height, width])
        Binary mask (1 for building interior, 0 for background & carved borders).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons:
        return mask

    if cv2 is not None:
        cv2_polys = []
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32)
            if len(pts) >= 3:
                cv2_polys.append(np.round(pts).astype(np.int32))

        if not cv2_polys:
            return mask

        # Step 1: Fill all polygons with 1
        cv2.fillPoly(mask, cv2_polys, 1)

        # Step 2: Carve borders with black outlines (0)
        boundary_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(
            boundary_mask,
            cv2_polys,
            isClosed=True,
            color=1,
            thickness=border_px,
        )
        mask[boundary_mask == 1] = 0

        # Step 3: Safeguard against completely eroded small buildings
        for poly in cv2_polys:
            # Check if this polygon has any remaining foreground pixels
            single_bldg = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(single_bldg, [poly], 1)
            if np.count_nonzero(mask & single_bldg) == 0:
                # Retain the centroid pixel so small buildings are never omitted
                cx = int(np.clip(np.mean(poly[:, 0]), 0, width - 1))
                cy = int(np.clip(np.mean(poly[:, 1]), 0, height - 1))
                mask[cy, cx] = 1
    else:
        # Fallback using PIL ImageDraw if cv2 is not yet ready
        from PIL import ImageDraw
        pil_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(pil_img)
        for poly in polygons:
            pts = [(p[0], p[1]) for p in poly]
            if len(pts) >= 3:
                draw.polygon(pts, fill=1)
                draw.line(pts + [pts[0]], fill=0, width=border_px)
        mask = np.array(pil_img, dtype=np.uint8)

    return mask


def load_label_polygons(label_path: Union[str, Path]) -> List[List[List[float]]]:
    """Extract all building polygon coordinates from a ground-truth label JSON."""
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygons: List[List[List[float]]] = []
    features = data.get("features", {}).get("xy", [])
    for feat in features:
        wkt = feat.get("wkt", "")
        if wkt:
            try:
                poly = parse_wkt_polygon(wkt)
                if len(poly) >= 3:
                    polygons.append(poly)
            except Exception:
                continue
    return polygons


class Stage1BuildingDataset(Dataset):
    """PyTorch Dataset for Stage-1 building localization.

    Loads pre-disaster images and rasterizes carved building masks.
    In training mode, takes random 512x512 native crops with geometric augmentations.
    In validation mode, evaluates on full 1024x1024 images.
    """

    def __init__(
        self,
        pair_ids: Sequence[str],
        data_root: Union[str, Path] = "Dataset/CIT_Hackathon_PS04_aerial/train",
        is_train: bool = True,
        crop_size: int = 512,
        border_px: int = 2,
        transform: Optional[Callable] = None,
    ):
        self.pair_ids = list(pair_ids)
        self.data_root = Path(data_root)
        self.is_train = is_train
        self.crop_size = crop_size
        self.border_px = border_px
        self.transform = transform

        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        pair_id = self.pair_ids[idx]
        img_path = self.images_dir / f"{pair_id}_pre.jpg"
        label_path = self.labels_dir / f"{pair_id}.json"

        # Load RGB image
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img, dtype=np.uint8)  # [1024, 1024, 3]

        # Rasterize boundary-carved mask
        polys = load_label_polygons(label_path)
        mask = rasterize_boundary_carved_mask(
            polys,
            height=img.shape[0],
            width=img.shape[1],
            border_px=self.border_px,
        )  # [1024, 1024] uint8 (0 or 1)

        # Training augmentations
        if self.is_train:
            h, w = img.shape[:2]
            cs = self.crop_size
            if h >= cs and w >= cs:
                max_y = h - cs
                max_x = w - cs
                cy = np.random.randint(0, max_y + 1)
                cx = np.random.randint(0, max_x + 1)
                img = img[cy : cy + cs, cx : cx + cs]
                mask = mask[cy : cy + cs, cx : cx + cs]

            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            # Random vertical flip
            if np.random.rand() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)

            # Random 90-degree rotations
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k)
                mask = np.rot90(mask, k)

        # Normalize image to float32 [0, 1] then apply ImageNet standardization
        img_norm = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.copy()).permute(2, 0, 1).float()  # [3, H, W]

        # Mask to tensor [1, H, W]
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).float()  # [1, H, W]

        return img_tensor, mask_tensor, pair_id


def get_stage1_dataloaders(
    splits_path: Union[str, Path] = "artifacts/splits.json",
    data_root: Union[str, Path] = "Dataset/CIT_Hackathon_PS04_aerial/train",
    batch_size: int = 4,
    val_batch_size: int = 1,
    num_workers: Optional[int] = None,
    crop_size: int = 512,
    border_px: int = 2,
) -> Tuple[Stage1BuildingDataset, Stage1BuildingDataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Factory creating train and validation Datasets and DataLoaders."""
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4
    splits_path = Path(splits_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = splits.get("train_pair_ids", [])
    val_ids = splits.get("val_pair_ids", [])

    train_ds = Stage1BuildingDataset(
        pair_ids=train_ids,
        data_root=data_root,
        is_train=True,
        crop_size=crop_size,
        border_px=border_px,
    )

    val_ds = Stage1BuildingDataset(
        pair_ids=val_ids,
        data_root=data_root,
        is_train=False,
        border_px=border_px,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_ds, val_ds, train_loader, val_loader
