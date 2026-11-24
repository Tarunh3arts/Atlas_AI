"""
Ground-truth label converter and event-based train/val splitter.

Public API
----------
``gt_label_to_scene_json``  – parse any label JSON → SceneJSON
``create_event_split``      – disaster-stratified 80/20 split with stats
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Union

from .schema import (
    BuildingInfo,
    DAMAGE_LEVELS,
    DAMAGE_NORMALIZE,
    IMAGE_MID_X,
    IMAGE_MID_Y,
    SceneJSON,
)


# ═══════════════════════════════════════════════════════════════
#  WKT POLYGON parser (no shapely dependency)
# ═══════════════════════════════════════════════════════════════
_WKT_POLYGON_RE = re.compile(
    r"POLYGON\s*\(\(\s*([\d\s.,eE+\-]+)\s*\)\)",
    re.IGNORECASE,
)


def parse_wkt_polygon(wkt: str) -> list[list[float]]:
    """Parse ``POLYGON ((x y, x y, …))`` → ``[[x, y], …]``."""
    m = _WKT_POLYGON_RE.match(wkt.strip())
    if not m:
        raise ValueError(f"Cannot parse WKT: {wkt[:120]!r}")
    points: list[list[float]] = []
    for pair in m.group(1).split(","):
        x_str, y_str = pair.strip().split()
        points.append([float(x_str), float(y_str)])
    return points


# ═══════════════════════════════════════════════════════════════
#  Pure-Python geometry helpers (no external deps)
# ═══════════════════════════════════════════════════════════════

def _remove_closing_duplicate(pts: list[list[float]]) -> list[list[float]]:
    """Strip the repeated closing vertex if present."""
    if len(pts) >= 2 and pts[0][0] == pts[-1][0] and pts[0][1] == pts[-1][1]:
        return pts[:-1]
    return pts


def polygon_area(pts: list[list[float]]) -> float:
    """Shoelace formula → unsigned area in px²."""
    pts = _remove_closing_duplicate(pts)
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) * 0.5


def polygon_centroid(pts: list[list[float]]) -> list[float]:
    """Polygon centroid computed as coordinate mean matching PS04 annotation conventions."""
    if not pts:
        return [0.0, 0.0]
    n = len(pts)
    return [
        sum(p[0] for p in pts) / n,
        sum(p[1] for p in pts) / n,
    ]


def polygon_bbox(pts: list[list[float]]) -> list[float]:
    """Axis-aligned bounding box → ``[x_min, y_min, x_max, y_max]``."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


# ── Spatial classification ──────────────────────────────────────

def classify_quadrant(cx: float, cy: float) -> str:
    """Return the quadrant name that contains ``(cx, cy)``."""
    v = "top" if cy < IMAGE_MID_Y else "bottom"
    h = "left" if cx < IMAGE_MID_X else "right"
    return f"{v}-{h}"


def classify_halves(cx: float, cy: float) -> list[str]:
    """Return ``[horizontal_half, vertical_half]``, e.g. ``['top', 'left']``."""
    return [
        "top" if cy < IMAGE_MID_Y else "bottom",
        "left" if cx < IMAGE_MID_X else "right",
    ]


# ═══════════════════════════════════════════════════════════════
#  gt_label_to_scene_json
# ═══════════════════════════════════════════════════════════════

def gt_label_to_scene_json(
    label_path: Union[str, Path],
    pair_id: str,
    pre_image_path: str,
    post_image_path: str,
) -> SceneJSON:
    """Convert a single ground-truth label JSON into a :class:`SceneJSON`.

    Parameters
    ----------
    label_path
        Path to ``{pair_id}.json`` inside ``train/labels/`` or ``practice/labels/``.
    pair_id
        The pair identifier (must match the label file stem).
    pre_image_path, post_image_path
        Relative or absolute paths to the pre- and post-disaster images.

    Returns
    -------
    SceneJSON
        Fully populated scene with all buildings, their spatial attributes,
        and normalized damage classes.  ``source`` is set to ``"ground_truth"``
        and every building has ``confidence = 1.0``.
    """
    label_path = Path(label_path)
    with open(label_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    metadata = raw.get("metadata", {})
    features = raw.get("features", {}).get("xy", [])

    buildings: list[BuildingInfo] = []
    for feat in features:
        props = feat["properties"]
        wkt   = feat["wkt"]

        polygon  = parse_wkt_polygon(wkt)
        bbox     = polygon_bbox(polygon)
        center   = polygon_centroid(polygon)
        area     = polygon_area(polygon)
        quadrant = classify_quadrant(center[0], center[1])
        halves   = classify_halves(center[0], center[1])

        raw_damage = props.get("subtype", "no-damage")
        damage     = DAMAGE_NORMALIZE.get(raw_damage, "no-damage")

        buildings.append(BuildingInfo(
            id         = props["uid"],
            polygon    = polygon,
            bbox       = [round(v, 2) for v in bbox],
            center     = [round(center[0], 2), round(center[1], 2)],
            area_px    = round(area, 1),
            quadrant   = quadrant,
            halves     = halves,
            damage     = damage,
            confidence = 1.0,
        ))

    return SceneJSON(
        pair_id         = pair_id,
        pre_image_path  = pre_image_path,
        post_image_path = post_image_path,
        image_size      = [metadata.get("width", 1024), metadata.get("height", 1024)],
        disaster        = metadata.get("disaster", ""),
        disaster_type   = metadata.get("disaster_type", ""),
        buildings       = buildings,
        source          = "ground_truth",
    )


# ═══════════════════════════════════════════════════════════════
#  Event-based train / val split
# ═══════════════════════════════════════════════════════════════

def _read_pairs_jsonl(path: Path) -> list[dict]:
    """Read ``pairs.jsonl`` → list of dicts."""
    pairs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _damage_stats_for_pairs(
    pair_ids: list[str],
    labels_dir: Path,
) -> dict:
    """Aggregate damage-class counts across the given pair IDs."""
    counter: Counter = Counter()
    for pid in pair_ids:
        label_file = labels_dir / f"{pid}.json"
        if not label_file.exists():
            continue
        with open(label_file, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for feat in raw["features"]["xy"]:
            sub = feat["properties"].get("subtype", "no-damage")
            counter[DAMAGE_NORMALIZE.get(sub, "no-damage")] += 1
    total = sum(counter.values())
    return {
        "total_buildings": total,
        "counts": {lvl: counter.get(lvl, 0) for lvl in DAMAGE_LEVELS},
        "percentages": {
            lvl: round(100.0 * counter.get(lvl, 0) / max(total, 1), 2)
            for lvl in DAMAGE_LEVELS
        },
    }


def create_event_split(
    pairs_jsonl_path: Union[str, Path],
    labels_dir: Union[str, Path],
    output_path: Union[str, Path],
    val_ratio: float = 0.20,
) -> dict:
    """Create an 80/20 split where **no disaster event leaks across splits**.

    The function tries every combination of 1‥4 held-out events and picks
    the one whose val-set size is closest to ``val_ratio`` of total pairs.

    Parameters
    ----------
    pairs_jsonl_path
        Path to ``train/pairs.jsonl``.
    labels_dir
        Path to ``train/labels/``.
    output_path
        Where to write the resulting ``splits.json``.
    val_ratio
        Target fraction for the validation set (default 0.20).

    Returns
    -------
    dict
        The splits dictionary (also saved as JSON).
    """
    pairs_jsonl_path = Path(pairs_jsonl_path)
    labels_dir       = Path(labels_dir)
    output_path      = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = _read_pairs_jsonl(pairs_jsonl_path)
    total = len(all_pairs)
    target_val = round(total * val_ratio)

    # Group by disaster event
    pairs_by_event: dict[str, list[dict]] = defaultdict(list)
    for p in all_pairs:
        pairs_by_event[p["disaster"]].append(p)
    events = sorted(pairs_by_event.keys())
    event_sizes = {e: len(pairs_by_event[e]) for e in events}

    # Brute-force best combination (8 events → ≤ 254 combos, instant)
    best_combo: tuple[str, ...] | None = None
    best_diff = float("inf")
    for r in range(1, len(events)):
        for combo in combinations(events, r):
            val_count = sum(event_sizes[e] for e in combo)
            diff = abs(val_count - target_val)
            if diff < best_diff:
                best_diff = diff
                best_combo = combo
        # Early exit if exact match found
        if best_diff == 0:
            break

    assert best_combo is not None
    val_events   = set(best_combo)
    train_events = set(events) - val_events

    train_ids = [p["pair_id"] for e in sorted(train_events) for p in pairs_by_event[e]]
    val_ids   = [p["pair_id"] for e in sorted(val_events)   for p in pairs_by_event[e]]

    result = {
        "train_events": sorted(train_events),
        "val_events":   sorted(val_events),
        "train_pair_ids": train_ids,
        "val_pair_ids":   val_ids,
        "train_count": len(train_ids),
        "val_count":   len(val_ids),
        "split_ratio": (
            f"{len(train_ids)}/{len(val_ids)} "
            f"({100 * len(train_ids) / total:.0f}%/"
            f"{100 * len(val_ids)   / total:.0f}%)"
        ),
        "train_damage_stats": _damage_stats_for_pairs(train_ids, labels_dir),
        "val_damage_stats":   _damage_stats_for_pairs(val_ids,   labels_dir),
        "event_summary": {
            e: {
                "pairs":    event_sizes[e],
                "split":    "val" if e in val_events else "train",
            }
            for e in events
        },
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return result
