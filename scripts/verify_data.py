"""
Verification script: parse 100% of practice + train labels into SceneJSON,
run the event-based split, and print summary stats.
"""

import json
import sys
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import gt_label_to_scene_json, create_event_split
from src.schema import DAMAGE_LEVELS, SceneJSON

BASE = PROJECT_ROOT / "Dataset" / "CIT_Hackathon_PS04_aerial"


def verify_split(split_name: str, labels_dir: Path, pairs_source: str):
    """Parse every label in a split and report results."""
    label_files = sorted(labels_dir.glob("*.json"))
    total = len(label_files)
    ok = 0
    fail = 0
    total_buildings = 0
    errors = []

    for lf in label_files:
        pair_id = lf.stem
        pre_img = f"images/{pair_id}_pre.jpg"
        post_img = f"images/{pair_id}_post.jpg"
        try:
            scene = gt_label_to_scene_json(lf, pair_id, pre_img, post_img)
            # Validate it round-trips through JSON
            json_str = scene.model_dump_json()
            reconstructed = SceneJSON.model_validate_json(json_str)
            assert len(reconstructed.buildings) == len(scene.buildings)
            # Check every building has valid fields
            for b in scene.buildings:
                assert b.quadrant in ("top-left", "top-right", "bottom-left", "bottom-right"), \
                    f"Bad quadrant: {b.quadrant}"
                assert len(b.halves) == 2, f"Bad halves: {b.halves}"
                assert b.halves[0] in ("top", "bottom"), f"Bad h-half: {b.halves[0]}"
                assert b.halves[1] in ("left", "right"), f"Bad v-half: {b.halves[1]}"
                assert b.damage in DAMAGE_LEVELS, f"Bad damage: {b.damage}"
                assert b.area_px >= 0, f"Negative area: {b.area_px}"
                assert len(b.bbox) == 4
                assert len(b.center) == 2
                assert len(b.polygon) >= 3, f"Degenerate polygon: {len(b.polygon)} pts"
            total_buildings += len(scene.buildings)
            ok += 1
        except Exception as e:
            fail += 1
            errors.append((pair_id, str(e)))
            traceback.print_exc()

    pct = 100.0 * ok / max(total, 1)
    print(f"\n{'='*60}")
    print(f"  {split_name}: {ok}/{total} labels parsed OK ({pct:.0f}%)")
    print(f"  Total buildings: {total_buildings}")
    if errors:
        print(f"  ERRORS ({fail}):")
        for pid, err in errors[:5]:
            print(f"    {pid}: {err}")
    print(f"{'='*60}")
    return ok, total, fail


def verify_scene_sample(split_dir: Path):
    """Print a sample SceneJSON to show the shape."""
    labels_dir = split_dir / "labels"
    first = sorted(labels_dir.glob("*.json"))[0]
    pair_id = first.stem
    scene = gt_label_to_scene_json(
        first, pair_id,
        f"images/{pair_id}_pre.jpg",
        f"images/{pair_id}_post.jpg",
    )
    print(f"\n--- Sample SceneJSON ({pair_id}) ---")
    print(f"pair_id:    {scene.pair_id}")
    print(f"disaster:   {scene.disaster} ({scene.disaster_type})")
    print(f"buildings:  {len(scene.buildings)}")
    if scene.buildings:
        b = scene.buildings[0]
        print(f"  [0] id={b.id}  damage={b.damage}  quadrant={b.quadrant}  "
              f"halves={b.halves}  area={b.area_px}  center={b.center}")
    print()


def main():
    print("=" * 60)
    print("  PS04 Dataset Verification")
    print("=" * 60)

    # 1. Parse all practice labels
    p_ok, p_total, p_fail = verify_split(
        "practice",
        BASE / "practice" / "labels",
        "practice",
    )

    # 2. Parse all train labels
    t_ok, t_total, t_fail = verify_split(
        "train",
        BASE / "train" / "labels",
        "train",
    )

    # 3. Sample output
    verify_scene_sample(BASE / "practice")

    # 4. Event-based split
    print("\n--- Creating event-based train/val split ---")
    artifacts_dir = PROJECT_ROOT / "artifacts"
    split_result = create_event_split(
        BASE / "train" / "pairs.jsonl",
        BASE / "train" / "labels",
        artifacts_dir / "splits.json",
    )
    print(f"Split ratio:    {split_result['split_ratio']}")
    print(f"Train events:   {split_result['train_events']}")
    print(f"Val events:     {split_result['val_events']}")
    print(f"Train pairs:    {split_result['train_count']}")
    print(f"Val pairs:      {split_result['val_count']}")
    print("\nTrain damage distribution:")
    for lvl in DAMAGE_LEVELS:
        cnt = split_result["train_damage_stats"]["counts"].get(lvl, 0)
        pct = split_result["train_damage_stats"]["percentages"].get(lvl, 0)
        print(f"  {lvl:16s}: {cnt:6d}  ({pct:.1f}%)")
    print("Val damage distribution:")
    for lvl in DAMAGE_LEVELS:
        cnt = split_result["val_damage_stats"]["counts"].get(lvl, 0)
        pct = split_result["val_damage_stats"]["percentages"].get(lvl, 0)
        print(f"  {lvl:16s}: {cnt:6d}  ({pct:.1f}%)")
    print(f"\nSplit saved to: {artifacts_dir / 'splits.json'}")

    # 5. Validate against practice/answers.jsonl schema
    print("\n--- Validating AnswerResponse schema against practice answers ---")
    from src.schema import AnswerResponse
    answers_path = BASE / "practice" / "answers.jsonl"
    ans_ok = 0
    with open(answers_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            # Build an AnswerResponse from practice gold
            ar = AnswerResponse(
                id=obj["id"],
                pair_id=obj["pair_id"],
                question_type=obj["type"],
                question=obj["question"],
                answer=obj["answer"],
                region=obj.get("region"),
                region_name=obj.get("region_name"),
                answerable=obj["answer"] != "cannot be determined",
            )
            sub = ar.to_submission()
            assert "id" in sub and "answer" in sub and "region" in sub
            ans_ok += 1
    print(f"  {ans_ok}/50 practice answers validated against AnswerResponse schema ✓")

    # Final summary
    all_ok = (p_fail == 0 and t_fail == 0)
    print(f"\n{'='*60}")
    if all_ok:
        print("  ✅ ALL CHECKS PASSED")
    else:
        print(f"  ❌ FAILURES: practice={p_fail}, train={t_fail}")
    print(f"{'='*60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
