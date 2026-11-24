"""
Stage 1 Inference: Full 1024x1024 building instance localization pipeline.

Steps:
1. Model forward pass on pre-disaster image (with optional Flip TTA).
2. Thresholding -> Euclidean Distance Transform -> Marker-based Watershed -> Min Area Filter.
3. Polygon extraction -> Centroid & Quadrant assignment -> BuildingInfo objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
except ImportError:
    peak_local_max = None
    watershed = None

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

from .data_utils import classify_halves, classify_quadrant, polygon_area, polygon_centroid
from .schema import BuildingInfo, SceneJSON
from .stage1_dataset import IMAGENET_MEAN, IMAGENET_STD


class Stage1Detector:
    """Full inference detector for pre-disaster building instance localization."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = "artifacts/stage1_config.json",
        encoder: str = "resnet34",
        device: Optional[str] = None,
        mask_threshold: Optional[float] = None,
        min_distance: Optional[int] = None,
        min_area_px: Optional[float] = None,
        use_tta: bool = False,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder = encoder
        self.use_tta = use_tta

        # Load tuned hyperparameters from artifacts/stage1_config.json by default
        cfg_thresh = 0.25
        cfg_dist = 14
        cfg_area = 35.0

        if config_path is not None:
            cp = Path(config_path)
            if cp.exists():
                try:
                    with open(cp, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    bp = cfg.get("best_params", {})
                    cfg_thresh = bp.get("mask_threshold", cfg_thresh)
                    cfg_dist = bp.get("min_distance", cfg_dist)
                    cfg_area = bp.get("min_area_px", cfg_area)
                except Exception:
                    pass

        self.mask_threshold = mask_threshold if mask_threshold is not None else cfg_thresh
        self.min_distance = min_distance if min_distance is not None else cfg_dist
        self.min_area_px = min_area_px if min_area_px is not None else cfg_area

        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[Union[str, Path]]) -> Optional[torch.nn.Module]:
        """Load SMP Unet model with weights if checkpoint exists."""
        if smp is None:
            return None

        model = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                state_dict = (
                    checkpoint["model_state_dict"]
                    if "model_state_dict" in checkpoint
                    else checkpoint
                )
                model.load_state_dict(state_dict)
                print(f"Loaded Stage-1 checkpoint from: {ckpt_path}")

        model = model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, img_input: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess 1024x1024 image into normalized tensor [1, 3, 1024, 1024]."""
        if isinstance(img_input, (str, Path)):
            pil_img = Image.open(img_input).convert("RGB")
            img = np.array(pil_img, dtype=np.uint8)
        elif isinstance(img_input, Image.Image):
            img = np.array(img_input.convert("RGB"), dtype=np.uint8)
        else:
            img = img_input

        img_norm = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_probability_map(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Run forward pass with optional test-time augmentation (TTA)."""
        if self.model is None:
            raise RuntimeError("Stage 1 model is not initialized.")

        logits = self.model(img_tensor)
        prob = torch.sigmoid(logits)

        if self.use_tta:
            # Horizontal flip TTA
            img_hflip = torch.flip(img_tensor, dims=[3])
            prob_hflip = torch.sigmoid(self.model(img_hflip))
            prob_hflip = torch.flip(prob_hflip, dims=[3])

            # Vertical flip TTA
            img_vflip = torch.flip(img_tensor, dims=[2])
            prob_vflip = torch.sigmoid(self.model(img_vflip))
            prob_vflip = torch.flip(prob_vflip, dims=[2])

            prob = (prob + prob_hflip + prob_vflip) / 3.0

        prob_map = prob.squeeze().cpu().numpy()  # [1024, 1024]
        return prob_map

    def extract_instances_watershed(
        self,
        prob_map: np.ndarray,
        mask_threshold: Optional[float] = None,
        min_distance: Optional[int] = None,
        min_area_px: Optional[float] = None,
    ) -> List[BuildingInfo]:
        """Convert probability map to BuildingInfo instances via Euclidean distance transform + watershed."""
        m_thresh = mask_threshold if mask_threshold is not None else self.mask_threshold
        m_dist = min_distance if min_distance is not None else self.min_distance
        m_area = min_area_px if min_area_px is not None else self.min_area_px

        # 1. Threshold probability map
        binary_mask = (prob_map >= m_thresh).astype(np.uint8)
        if np.count_nonzero(binary_mask) == 0:
            return []

        # 2. Euclidean Distance Transform
        distance = ndi.distance_transform_edt(binary_mask)

        # 3. Peak / marker detection
        if peak_local_max is not None:
            coords = peak_local_max(
                distance,
                min_distance=max(1, m_dist),
                labels=binary_mask,
                exclude_border=False,
            )
            peak_mask = np.zeros(distance.shape, dtype=bool)
            if len(coords) > 0:
                peak_mask[tuple(coords.T)] = True
            markers, num_markers = ndi.label(peak_mask)
        else:
            footprint = 2 * max(1, m_dist) + 1
            local_max = ndi.maximum_filter(distance, size=footprint) == distance
            peaks = local_max & (distance > 1.0) & (binary_mask > 0)
            markers, num_markers = ndi.label(peaks)

        # Fallback to connected components if no peaks were isolated
        if num_markers == 0:
            markers, num_markers = ndi.label(binary_mask)

        # 4. Marker-based watershed
        if watershed is not None:
            labeled_instances = watershed(-distance, markers, mask=binary_mask)
        else:
            # Fallback connected components if skimage watershed unavailable
            labeled_instances = markers

        # 5. Extract building instances
        unique_labels = np.unique(labeled_instances)
        buildings: List[BuildingInfo] = []
        bldg_idx = 0

        for lbl in unique_labels:
            if lbl == 0:
                continue

            inst_mask = (labeled_instances == lbl).astype(np.uint8)
            inst_area = float(np.count_nonzero(inst_mask))
            if inst_area < m_area:
                continue

            # Extract contours
            if cv2 is not None:
                contours, _ = cv2.findContours(
                    inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue
                c = max(contours, key=cv2.contourArea)
                if len(c) < 3:
                    continue
                poly = [[float(pt[0][0]), float(pt[0][1])] for pt in c]
            else:
                # Fallback bbox polygon if cv2 unavailable
                ys, xs = np.where(inst_mask)
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]

            # Coordinates, centroid, bbox
            poly_xs = [p[0] for p in poly]
            poly_ys = [p[1] for p in poly]
            bbox = [
                float(min(poly_xs)),
                float(min(poly_ys)),
                float(max(poly_xs)),
                float(max(poly_ys)),
            ]
            bw = max(bbox[2] - bbox[0], 1.0)
            bh = max(bbox[3] - bbox[1], 1.0)
            aspect_ratio = bw / bh
            if aspect_ratio < 0.15 or aspect_ratio > 6.5:
                # Reject extreme aspect-ratio geometric slivers
                continue

            center = polygon_centroid(poly)
            calc_area = polygon_area(poly)
            area_val = calc_area if calc_area > 0 else inst_area
            if area_val < m_area:
                continue

            # Mean probability inside the building
            conf = float(np.mean(prob_map[inst_mask == 1]))

            cx, cy = center[0], center[1]
            quadrant = classify_quadrant(cx, cy)
            halves = classify_halves(cx, cy)

            buildings.append(
                BuildingInfo(
                    id=f"det_{bldg_idx}",
                    polygon=poly,
                    bbox=[round(v, 2) for v in bbox],
                    center=[round(cx, 2), round(cy, 2)],
                    area_px=round(area_val, 1),
                    quadrant=quadrant,
                    halves=halves,
                    damage="no-damage",  # Stage 2 Siamese classifier will update this
                    confidence=round(conf, 4),
                )
            )
            bldg_idx += 1

        return buildings

    def detect(
        self,
        img_path: Union[str, Path],
        pair_id: str = "custom",
        post_image_path: Optional[str] = None,
        mask_threshold: Optional[float] = None,
        min_distance: Optional[int] = None,
        min_area_px: Optional[float] = None,
    ) -> SceneJSON:
        """Run full detection pipeline on a pre-disaster image and return a SceneJSON."""
        img_tensor = self.preprocess_image(img_path)
        prob_map = self.predict_probability_map(img_tensor)
        buildings = self.extract_instances_watershed(
            prob_map,
            mask_threshold=mask_threshold,
            min_distance=min_distance,
            min_area_px=min_area_px,
        )

        return SceneJSON(
            pair_id=pair_id,
            pre_image_path=str(img_path),
            post_image_path=str(post_image_path or img_path),
            image_size=[1024, 1024],
            buildings=buildings,
            source="prediction",
        )
