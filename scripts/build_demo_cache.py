"""
Pre-caches ground-truth scenes and sample Q&A for instant live Streamlit demo fallback.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE = PROJECT_ROOT / "Dataset" / "CIT_Hackathon_PS04_aerial" / "practice"

from src.data_utils import gt_label_to_scene_json
from src.qa_parser import parse_question
from src.spatial_engine import answer_question
from src.schema import SceneJSON


def build_demo_cache():
    sample_pairs = [
        "p11b4ad02",
        "p4ca216d8",
        "p78d825ba",
        "pfd111001",
        "p57d3d177",
        "p6adc4d5d",
        "pb67aa0fe",
        "pc9ad52c9",
    ]

    cache = {}
    for pid in sample_pairs:
        label_p = BASE / f"labels/{pid}.json"
        pre_p = f"Dataset/CIT_Hackathon_PS04_aerial/practice/images/{pid}_pre.jpg"
        post_p = f"Dataset/CIT_Hackathon_PS04_aerial/practice/images/{pid}_post.jpg"

        scene = gt_label_to_scene_json(label_p, pid, pre_p, post_p)
        cache[pid] = {
            "pair_id": pid,
            "disaster": scene.disaster,
            "disaster_type": scene.disaster_type,
            "pre_image": pre_p,
            "post_image": post_p,
            "buildings": [b.model_dump() for b in scene.buildings],
            "total_buildings": len(scene.buildings),
            "damage_counts": {
                "no-damage": sum(1 for b in scene.buildings if b.damage == "no-damage"),
                "minor-damage": sum(1 for b in scene.buildings if b.damage == "minor-damage"),
                "major-damage": sum(1 for b in scene.buildings if b.damage == "major-damage"),
                "destroyed": sum(1 for b in scene.buildings if b.damage == "destroyed"),
            },
        }

    out_file = PROJECT_ROOT / "artifacts" / "demo_cache.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    print(f"Pre-cached {len(cache)} demo pairs to {out_file}")


if __name__ == "__main__":
    build_demo_cache()
