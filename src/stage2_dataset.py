"""
Stage 2 Dataset: Paired Pre/Post Building Crop Loader with Box Jitter & Balanced Sampling.

Key Features:
1. Extracts 1.3x padded crops around building bounding boxes from pre & post images, resized to 96x96.
2. Applies bounding-box shift (±10%) and scale jitter during training to ensure robustness against
   imperfect Stage-1 detections and minor satellite coregistration errors.
3. Synchronized geometric augmentations (random horizontal/vertical flips and 90-degree rotations).
4. WeightedRandomSampler integration to handle severe class imbalance across the 4 damage tiers:
   no-damage (86%), minor-damage (3%), major-damage (4%), destroyed (5%).
5. Excludes 'un-classified' buildings from training supervision.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

from .data_utils import parse_wkt_polygon, polygon_bbox
from .schema import DAMAGE_LEVELS, DAMAGE_NORMALIZE, DamageLevel

# Damage class index mapping
DAMAGE_TO_IDX: Dict[str, int] = {
    DamageLevel.NO_DAMAGE.value: 0,
    DamageLevel.MINOR_DAMAGE.value: 1,
    DamageLevel.MAJOR_DAMAGE.value: 2,
    DamageLevel.DESTROYED.value: 3,
}

IDX_TO_DAMAGE: Dict[int, str] = {v: k for k, v in DAMAGE_TO_IDX.items()}

# Standard ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_padded_crop(
    image: np.ndarray,
    bbox: Sequence[float],
    pad_factor: float = 1.3,
    crop_size: int = 96,
    jitter: bool = False,
    jitter_shift_pct: float = 0.10,
    jitter_scale_range: Tuple[float, float] = (0.90, 1.10),
) -> np.ndarray:
    """Extract a padded square crop centered on the bounding box with optional shift/scale jitter.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape [H, W, 3].
    bbox : Sequence[float]
        Bounding box [x1, y1, x2, y2].
    pad_factor : float
        Padding multiplier relative to the bounding box dimensions (default 1.3).
    crop_size : int
        Target width and height of the resized crop (default 96).
    jitter : bool
        Whether to apply random shift and scale perturbations (for training).
    jitter_shift_pct : float
        Maximum fraction of crop size to randomly shift center (default 0.10 = ±10%).
    jitter_scale_range : Tuple[float, float]
        Range of random scale multipliers (default 0.90 to 1.10).

    Returns
    -------
    np.ndarray
        RGB crop of shape [crop_size, crop_size, 3] uint8.
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Square side length covering 1.3x of max dimension
    side = max(w, h) * pad_factor
    side = max(side, 12.0)  # Minimum side to avoid sub-pixel degenerate crops

    if jitter:
        # Random scale jitter (±10%)
        scale = np.random.uniform(jitter_scale_range[0], jitter_scale_range[1])
        side = side * scale

        # Random center shift (±10%)
        shift_x = np.random.uniform(-jitter_shift_pct, jitter_shift_pct) * side
        shift_y = np.random.uniform(-jitter_shift_pct, jitter_shift_pct) * side
        cx += shift_x
        cy += shift_y

    crop_x1 = int(round(cx - side / 2.0))
    crop_y1 = int(round(cy - side / 2.0))
    crop_x2 = int(round(cx + side / 2.0))
    crop_y2 = int(round(cy + side / 2.0))

    img_h, img_w = image.shape[:2]

    # Calculate padding needed if crop falls outside image boundaries
    pad_top = max(0, -crop_y1)
    pad_bottom = max(0, crop_y2 - img_h)
    pad_left = max(0, -crop_x1)
    pad_right = max(0, crop_x2 - img_w)

    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(img_w, crop_x2)
    src_y2 = min(img_h, crop_y2)

    cropped = image[src_y1:src_y2, src_x1:src_x2]

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if cv2 is not None:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REFLECT_101,
            )
        else:
            cropped = np.pad(
                cropped,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="reflect",
            )

    # Resize to target crop_size x crop_size
    if cv2 is not None:
        resized = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    else:
        pil_crop = Image.fromarray(cropped)
        resized = np.array(
            pil_crop.resize((crop_size, crop_size), Image.Resampling.BILINEAR)
        )

    return resized


class BuildingCropSample:
    """Lightweight metadata descriptor for a single building instance crop."""

    __slots__ = ("pair_id", "bldg_uid", "bbox", "damage_class", "label_idx")

    def __init__(
        self,
        pair_id: str,
        bldg_uid: str,
        bbox: List[float],
        damage_class: str,
        label_idx: int,
    ):
        self.pair_id = pair_id
        self.bldg_uid = bldg_uid
        self.bbox = bbox
        self.damage_class = damage_class
        self.label_idx = label_idx


class Stage2SiameseCropDataset(Dataset):
    """PyTorch Dataset loading paired pre & post building crops for damage tier classification."""

    def __init__(
        self,
        pair_ids: Sequence[str],
        data_root: Union[str, Path] = "Dataset/CIT_Hackathon_PS04_aerial/train",
        is_train: bool = True,
        crop_size: int = 96,
        pad_factor: float = 1.3,
        jitter_shift_pct: float = 0.10,
        jitter_scale_range: Tuple[float, float] = (0.90, 1.10),
        cache_images: bool = False,
        precache_crops: bool = True,
    ):
        self.data_root = Path(data_root)
        self.is_train = is_train
        self.crop_size = crop_size
        self.pad_factor = pad_factor
        self.jitter_shift_pct = jitter_shift_pct
        self.jitter_scale_range = jitter_scale_range
        self.cache_images = cache_images
        self.precache_crops = precache_crops

        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"

        # Image cache: pair_id -> (pre_img, post_img)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Index all valid building instances from the given pair_ids
        self.samples: List[BuildingCropSample] = self._index_buildings(pair_ids)

        # Pre-cache all 96x96 building crops in RAM (single disk read per 1024x1024 image pair)
        self.cached_pre: Optional[np.ndarray] = None
        self.cached_post: Optional[np.ndarray] = None
        self.cached_labels: Optional[np.ndarray] = None

        if self.precache_crops and len(self.samples) > 0:
            self._precache_all_crops()

    def _precache_all_crops(self):
        """Pre-cache all 96x96 building crops in RAM.
        Groups crops by image pair so each 1024x1024 image is opened from disk only ONCE at startup.
        Total RAM footprint: ~850 MB for ~31k crops."""
        n = len(self.samples)
        self.cached_pre = np.empty((n, self.crop_size, self.crop_size, 3), dtype=np.uint8)
        self.cached_post = np.empty((n, self.crop_size, self.crop_size, 3), dtype=np.uint8)
        self.cached_labels = np.empty((n,), dtype=np.int64)

        pair_to_indices = defaultdict(list)
        for idx, s in enumerate(self.samples):
            pair_to_indices[s.pair_id].append(idx)

        desc = f"Pre-caching {'train' if self.is_train else 'val'} crops in RAM"
        for pair_id, indices in tqdm(pair_to_indices.items(), desc=desc, leave=False):
            pre_path = self.images_dir / f"{pair_id}_pre.jpg"
            post_path = self.images_dir / f"{pair_id}_post.jpg"

            pre_full = np.array(Image.open(pre_path).convert("RGB"), dtype=np.uint8)
            post_full = np.array(Image.open(post_path).convert("RGB"), dtype=np.uint8)

            for idx in indices:
                s = self.samples[idx]
                c_pre = extract_padded_crop(
                    pre_full,
                    s.bbox,
                    pad_factor=self.pad_factor,
                    crop_size=self.crop_size,
                    jitter=False,
                )
                c_post = extract_padded_crop(
                    post_full,
                    s.bbox,
                    pad_factor=self.pad_factor,
                    crop_size=self.crop_size,
                    jitter=False,
                )
                self.cached_pre[idx] = c_pre
                self.cached_post[idx] = c_post
                self.cached_labels[idx] = s.label_idx

    def _index_buildings(self, pair_ids: Sequence[str]) -> List[BuildingCropSample]:
        """Scan ground truth labels and index all annotated buildings with valid damage tiers."""
        samples = []
        for pid in pair_ids:
            label_file = self.labels_dir / f"{pid}.json"
            if not label_file.exists():
                continue

            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for feat in data.get("features", {}).get("xy", []):
                raw_subtype = feat.get("properties", {}).get("subtype", "no-damage")
                # Skip un-classified buildings for Stage 2 supervised training
                if raw_subtype == "un-classified":
                    continue

                canonical_damage = DAMAGE_NORMALIZE.get(raw_subtype, "no-damage")
                if canonical_damage not in DAMAGE_TO_IDX:
                    continue

                wkt = feat.get("wkt", "")
                if not wkt:
                    continue

                try:
                    poly = parse_wkt_polygon(wkt)
                    if len(poly) < 3:
                        continue
                    bbox = polygon_bbox(poly)
                    samples.append(
                        BuildingCropSample(
                            pair_id=pid,
                            bldg_uid=feat.get("properties", {}).get("uid", f"b{len(samples)}"),
                            bbox=bbox,
                            damage_class=canonical_damage,
                            label_idx=DAMAGE_TO_IDX[canonical_damage],
                        )
                    )
                except Exception:
                    continue
        return samples

    def get_class_counts(self) -> Dict[str, int]:
        """Return sample counts per damage class."""
        counts = {k: 0 for k in DAMAGE_LEVELS}
        for s in self.samples:
            counts[s.damage_class] += 1
        return counts

    def get_sample_weights(self, power: float = 0.5) -> torch.DoubleTensor:
        """Calculate softened inverse class frequency weights for WeightedRandomSampler.

        Parameters
        ----------
        power : float
            Exponent for inverse frequency. power=0.5 corresponds to softened
            inverse square-root weighting (1.0 / sqrt(count)), balancing minority
            damage tiers while preventing severe over-prediction of damage on intact roofs.
        """
        counts = self.get_class_counts()
        total = len(self.samples)
        class_weights = {}
        for c, count in counts.items():
            if power == 0.5:
                # Softened inverse square-root weighting
                class_weights[DAMAGE_TO_IDX[c]] = 1.0 / np.sqrt(max(count, 1))
            elif power == 1.0:
                class_weights[DAMAGE_TO_IDX[c]] = total / max(count, 1)
            else:
                class_weights[DAMAGE_TO_IDX[c]] = (total / max(count, 1)) ** power

        weights = [class_weights[s.label_idx] for s in self.samples]
        return torch.DoubleTensor(weights)

    def _load_image_pair(self, pair_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load RGB pre and post images for a given pair ID."""
        if self.cache_images and pair_id in self._cache:
            return self._cache[pair_id]

        pre_path = self.images_dir / f"{pair_id}_pre.jpg"
        post_path = self.images_dir / f"{pair_id}_post.jpg"

        pre_img = np.array(Image.open(pre_path).convert("RGB"), dtype=np.uint8)
        post_img = np.array(Image.open(post_path).convert("RGB"), dtype=np.uint8)

        if self.cache_images:
            self._cache[pair_id] = (pre_img, post_img)

        return pre_img, post_img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        sample = self.samples[idx]

        if self.cached_pre is not None:
            crop_pre = self.cached_pre[idx].copy()
            crop_post = self.cached_post[idx].copy()
            label_idx = int(self.cached_labels[idx])
        else:
            pre_full, post_full = self._load_image_pair(sample.pair_id)
            crop_pre = extract_padded_crop(
                pre_full,
                sample.bbox,
                pad_factor=self.pad_factor,
                crop_size=self.crop_size,
                jitter=False,
            )
            crop_post = extract_padded_crop(
                post_full,
                sample.bbox,
                pad_factor=self.pad_factor,
                crop_size=self.crop_size,
                jitter=False,
            )
            label_idx = sample.label_idx

        # In training mode, apply shift jitter + synchronized flips/rotations
        if self.is_train:
            # Shift jitter: ±10% of 96px = ±9 pixels
            shift_x = int(round(np.random.uniform(-self.jitter_shift_pct, self.jitter_shift_pct) * self.crop_size))
            shift_y = int(round(np.random.uniform(-self.jitter_shift_pct, self.jitter_shift_pct) * self.crop_size))
            if shift_x != 0 or shift_y != 0:
                if cv2 is not None:
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    crop_pre = cv2.warpAffine(crop_pre, M, (self.crop_size, self.crop_size), borderMode=cv2.BORDER_REFLECT)
                    crop_post = cv2.warpAffine(crop_post, M, (self.crop_size, self.crop_size), borderMode=cv2.BORDER_REFLECT)
                else:
                    crop_pre = np.roll(np.roll(crop_pre, shift_y, axis=0), shift_x, axis=1)
                    crop_post = np.roll(np.roll(crop_post, shift_y, axis=0), shift_x, axis=1)

            # Synchronized horizontal flip
            if np.random.rand() > 0.5:
                crop_pre = np.fliplr(crop_pre).copy()
                crop_post = np.fliplr(crop_post).copy()

            # Synchronized vertical flip
            if np.random.rand() > 0.5:
                crop_pre = np.flipud(crop_pre).copy()
                crop_post = np.flipud(crop_post).copy()

            # Synchronized 90-degree rotations
            k = np.random.randint(0, 4)
            if k > 0:
                crop_pre = np.rot90(crop_pre, k).copy()
                crop_post = np.rot90(crop_post, k).copy()

        # ImageNet normalization [0, 1] -> standardized tensor
        pre_norm = (crop_pre.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        post_norm = (crop_post.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        pre_tensor = torch.from_numpy(pre_norm).permute(2, 0, 1).float()  # [3, 96, 96]
        post_tensor = torch.from_numpy(post_norm).permute(2, 0, 1).float()  # [3, 96, 96]

        return pre_tensor, post_tensor, label_idx, sample.bldg_uid


def get_stage2_dataloaders(
    splits_path: Union[str, Path] = "artifacts/splits.json",
    data_root: Union[str, Path] = "Dataset/CIT_Hackathon_PS04_aerial/train",
    batch_size: int = 64,
    val_batch_size: int = 64,
    num_workers: Optional[int] = None,
    crop_size: int = 96,
    pad_factor: float = 1.3,
    use_weighted_sampler: bool = True,
    sampler_power: float = 0.5,
    num_samples: Optional[int] = 12000,
    precache_crops: bool = True,
) -> Tuple[Stage2SiameseCropDataset, Stage2SiameseCropDataset, DataLoader, DataLoader]:
    """Factory creating balanced train and validation DataLoaders for Stage 2."""
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4

    splits_path = Path(splits_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = splits.get("train_pair_ids", [])
    val_ids = splits.get("val_pair_ids", [])

    train_ds = Stage2SiameseCropDataset(
        pair_ids=train_ids,
        data_root=data_root,
        is_train=True,
        crop_size=crop_size,
        pad_factor=pad_factor,
        precache_crops=precache_crops,
    )

    val_ds = Stage2SiameseCropDataset(
        pair_ids=val_ids,
        data_root=data_root,
        is_train=False,
        crop_size=crop_size,
        pad_factor=pad_factor,
        precache_crops=precache_crops,
    )

    if use_weighted_sampler and len(train_ds) > 0:
        eff_samples = num_samples if num_samples is not None and num_samples > 0 else len(train_ds)
        sampler = WeightedRandomSampler(
            weights=train_ds.get_sample_weights(power=sampler_power),
            num_samples=eff_samples,
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(len(train_ds) > 0),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_ds, val_ds, train_loader, val_loader
