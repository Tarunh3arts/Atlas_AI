"""
Benchmark: compare our QA engine (GT scene JSONs) against practice/answers.jsonl.

Prints per-type accuracy and details of every mismatch.
Exit code 0 iff all 50 answers match.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import gt_label_to_scene_json
from src.qa_parser import parse_question
from src.schema import SceneJSON
from src.spatial_engine import answer_question

BASE = PROJECT_ROOT / "Dataset" / "CIT_Hackathon_PS04_aerial"
PRACTICE = BASE / "practice"


def main() -> int:
    # ── Load gold answers ───────────────────────────────────────
    gold: dict[str, dict] = {}
    with open(PRACTICE / "answers.jsonl") as f:
        for line in f:
            obj = json.loads(line.strip())
            gold[obj["id"]] = obj

    # ── Load questions ──────────────────────────────────────────
    questions: list[dict] = []
    with open(PRACTICE / "questions.jsonl") as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    # ── Pre-build SceneJSON cache per pair_id ───────────────────
    scene_cache: dict[str, SceneJSON] = {}
    for q in questions:
        pid = q["pair_id"]
        if pid not in scene_cache:
            label_path = PRACTICE / "labels" / f"{pid}.json"
            scene_cache[pid] = gt_label_to_scene_json(
                label_path, pid,
                f"images/{pid}_pre.jpg",
                f"images/{pid}_post.jpg",
            )

    # ── Evaluate ────────────────────────────────────────────────
    type_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    type_total   = {1: 0, 2: 0, 3: 0, 4: 0}
    mismatches   = []

    for q in questions:
        qid   = q["id"]
        pid   = q["pair_id"]
        qtext = q["question"]
        g     = gold[qid]

        scene  = scene_cache[pid]
        parsed = parse_question(qtext)
        result = answer_question(scene, parsed, qid)

        g_type   = g["type"]
        g_answer = g["answer"]

        type_total[g_type] += 1

        # ── Compare answer ──────────────────────────────────────
        answer_ok = (result.answer == g_answer)
        if answer_ok:
            type_correct[g_type] += 1
        else:
            mismatches.append({
                "id":          qid,
                "pair_id":     pid,
                "question":    qtext,
                "gold_type":   g_type,
                "pred_type":   parsed.question_type,
                "gold_answer": g_answer,
                "pred_answer": result.answer,
                "gold_region": g.get("region"),
                "pred_region": result.region,
                "n_buildings": len(scene.buildings),
            })

    # ── Report ──────────────────────────────────────────────────
    total_correct = sum(type_correct.values())
    total         = sum(type_total.values())

    print("=" * 64)
    print(f"  Practice-Set Benchmark  —  {total_correct}/{total} correct")
    print("=" * 64)

    for t in (1, 2, 3, 4):
        c, n = type_correct[t], type_total[t]
        pct = 100.0 * c / n if n else 0
        status = "✅" if c == n else "❌"
        print(f"  Type {t}: {c:2d}/{n:2d}  ({pct:5.1f}%)  {status}")

    if mismatches:
        print(f"\n{'─'*64}")
        print(f"  {len(mismatches)} MISMATCH(ES):")
        print(f"{'─'*64}")
        for mm in mismatches:
            print(f"\n  [{mm['id']}]  pair={mm['pair_id']}  "
                  f"buildings={mm['n_buildings']}")
            print(f"    Q:    {mm['question']}")
            print(f"    Gold: type={mm['gold_type']}  answer={mm['gold_answer']!r}"
                  f"  region={mm['gold_region']}")
            print(f"    Pred: type={mm['pred_type']}  answer={mm['pred_answer']!r}"
                  f"  region={mm['pred_region']}")

    print(f"\n{'=' * 64}")
    if total_correct == total:
        print("  ✅  ALL 50 ANSWERS MATCH — 100% accuracy")
    else:
        print(f"  ❌  {total - total_correct} wrong — needs fixing")
    print("=" * 64)

    return 0 if total_correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
