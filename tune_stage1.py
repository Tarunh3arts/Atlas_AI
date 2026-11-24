"""
Stage 1 Post-Processing Hyperparameter Tuner.

Tunes:
  - mask_threshold: Probability threshold for foreground buildings
  - min_distance: Peak separation in Euclidean distance transform
  - min_area_px: Minimum polygon area to reject false positive noise

Objective:
  Directly minimizes Building Count Mean Absolute Error (MAE) across the
  held-out validation split.

Precomputes model probability maps first, then rapidly evaluates parameter
combinations in memory. Saves optimal configuration to artifacts/stage1_config.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
except ImportError:
    peak_local_max = None
    watershed = None

from src.data_utils import parse_wkt_polygon
from src.stage1_dataset import load_label_polygons
from src.stage1_infer import Stage1Detector


def count_instances_fast(
    prob_map: np.ndarray,
    mask_threshold: float,
    min_distance: int,
    min_area_px: float,
) -> int:
    """Ultra-fast building count via threshold + distance transform + watershed."""
    binary_mask = (prob_map >= mask_threshold).astype(np.uint8)
    if not binary_mask.any():
        return 0

    distance = ndi.distance_transform_edt(binary_mask)

    if peak_local_max is not None:
        coords = peak_local_max(
            distance,
            min_distance=max(1, min_distance),
            labels=binary_mask,
            exclude_border=False,
        )
        if len(coords) == 0:
            return 0
        peak_mask = np.zeros(distance.shape, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, num_markers = ndi.label(peak_mask)
    else:
        footprint = 2 * max(1, min_distance) + 1
        local_max = ndi.maximum_filter(distance, size=footprint) == distance
        peaks = local_max & (distance > 1.0) & (binary_mask > 0)
        markers, num_markers = ndi.label(peaks)

    if num_markers == 0:
        markers, num_markers = ndi.label(binary_mask)

    if watershed is not None:
        labeled = watershed(-distance, markers, mask=binary_mask)
    else:
        labeled = markers

    counts = np.bincount(labeled.ravel())[1:]
    if len(counts) == 0:
        return 0
    return int(np.count_nonzero(counts >= min_area_px))


def evaluate_single_config(
    args_tuple: Tuple[float, int, float, Dict[str, np.ndarray], Dict[str, int]]
) -> Tuple[float, int, float, float, float, float]:
    """Worker function evaluating one (mask_threshold, min_distance, min_area_px) tuple."""
    mask_threshold, min_distance, min_area_px, prob_maps, gt_counts = args_tuple
    errors = []
    rel_errors = []
    signed_diffs = []

    for pid, prob in prob_maps.items():
        gt = gt_counts.get(pid, 0)
        pred = count_instances_fast(
            prob,
            mask_threshold=mask_threshold,
            min_distance=min_distance,
            min_area_px=min_area_px,
        )
        abs_err = abs(pred - gt)
        errors.append(abs_err)
        signed_diffs.append(pred - gt)
        rel_errors.append(abs_err / max(gt, 1))

    mae = float(np.mean(errors))
    mape = float(np.mean(rel_errors)) * 100.0
    mean_bias = float(np.mean(signed_diffs))
    return mask_threshold, min_distance, min_area_px, mae, mape, mean_bias


def load_val_ground_truth(
    val_pair_ids: List[str], data_root: Path
) -> Dict[str, int]:
    """Load ground-truth building counts for validation pairs."""
    labels_dir = data_root / "labels"
    gt_counts = {}
    for pid in val_pair_ids:
        label_file = labels_dir / f"{pid}.json"
        polys = load_label_polygons(label_file)
        gt_counts[pid] = len(polys)
    return gt_counts


def precompute_probability_maps(
    detector: Stage1Detector,
    val_pair_ids: List[str],
    data_root: Path,
) -> Dict[str, np.ndarray]:
    """Precompute probability maps for all validation images."""
    images_dir = data_root / "images"
    prob_maps = {}
    print(f"Precomputing probability maps for {len(val_pair_ids)} validation images...")
    start = time.time()

    for idx, pid in enumerate(val_pair_ids):
        img_path = images_dir / f"{pid}_pre.jpg"
        if not img_path.exists():
            continue
        tensor = detector.preprocess_image(img_path)
        prob_map = detector.predict_probability_map(tensor)
        prob_maps[pid] = prob_map
        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_pair_ids):
            print(f"  Processed {idx + 1}/{len(val_pair_ids)} images ({time.time() - start:.1f}s)")

    return prob_maps


def main():
    parser = argparse.ArgumentParser(description="Tune Stage 1 Post-Processing Parameters")
    parser.add_argument("--checkpoint", type=str, default="artifacts/stage1_best.pth")
    parser.add_argument("--splits_file", type=str, default="artifacts/splits.json")
    parser.add_argument("--data_root", type=str, default="Dataset/CIT_Hackathon_PS04_aerial/train")
    parser.add_argument("--output_config", type=str, default="artifacts/stage1_config.json")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit val set size for rapid search")
    args = parser.parse_args()

    print("=" * 64)
    print("  PS04 Stage 1: Post-Processing Hyperparameter Tuner")
    print("=" * 64)

    data_root = Path(args.data_root)
    output_config = Path(args.output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Validation Split IDs
    with open(args.splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    val_ids = splits.get("val_pair_ids", [])
    if args.max_samples is not None and args.max_samples > 0:
        val_ids = val_ids[: args.max_samples]

    print(f"Loaded {len(val_ids)} validation pairs from {args.splits_file}.")

    # 2. Load GT counts
    gt_counts = load_val_ground_truth(val_ids, data_root)
    total_gt = sum(gt_counts.values())
    print(f"Total ground-truth buildings in tuning set: {total_gt} (mean: {total_gt / len(val_ids):.1f}/img)")

    # 3. Initialize Detector
    detector = Stage1Detector(
        checkpoint_path=args.checkpoint if Path(args.checkpoint).exists() else None,
        encoder=args.encoder,
        device=args.device,
    )

    # 4. Precompute Probability Maps
    prob_maps = precompute_probability_maps(detector, val_ids, data_root)

    # 5. Grid Search Grid
    mask_thresholds = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    min_distances = [6, 8, 10, 14, 18, 22]
    min_areas = [20.0, 35.0, 50.0, 80.0]

    combos = list(itertools.product(mask_thresholds, min_distances, min_areas))
    total_combos = len(combos)
    max_workers = min(16, os.cpu_count() or 4)
    print(f"\nStarting Fast Parallel Grid Search over {total_combos} combinations with {max_workers} threads...")

    best_mae = float("inf")
    best_params = {}
    best_metrics = {}

    grid_start = time.time()
    worker_args = [(t, d, a, prob_maps, gt_counts) for t, d, a in combos]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_config, arg) for arg in worker_args]
        for future in concurrent.futures.as_completed(futures):
            thresh, dist, area, mae, mape, bias = future.result()
            if mae < best_mae:
                best_mae = mae
                best_params = {
                    "mask_threshold": thresh,
                    "min_distance": dist,
                    "min_area_px": area,
                }
                best_metrics = {
                    "val_count_mae": round(mae, 3),
                    "val_count_mape": round(mape, 2),
                    "mean_bias": round(bias, 2),
                }
                print(
                    f"  ★ New Best -> MAE: {mae:6.2f} | MAPE: {mape:5.1f}% | Bias: {bias:+5.2f} "
                    f"| thresh={thresh:.2f}, min_dist={dist:2d}, min_area={area:.0f}px"
                )

    elapsed = time.time() - grid_start
    print("\n" + "=" * 64)
    print(f"Grid search completed in {elapsed:.1f}s.")
    print("Optimal Parameters:")
    for k, v in best_params.items():
        print(f"  {k:18s}: {v}")
    print("Validation Metrics:")
    for k, v in best_metrics.items():
        print(f"  {k:18s}: {v}")
    print(f"Saving config to: {output_config}")
    print("=" * 64)

    final_payload = {
        "encoder": args.encoder,
        "best_params": best_params,
        "validation_metrics": best_metrics,
        "evaluated_pairs": len(val_ids),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)


if __name__ == "__main__":
    import torch

    main()
