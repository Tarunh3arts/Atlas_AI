"""
Comprehensive End-to-End Evaluation Harness for PS04 Disaster Visual Question-Answering.

Evaluates the FULL model pipeline (Stage 1 localization + Stage 2 Siamese damage
classification + Stage 3 spatial QA) WITHOUT using ground-truth labels:

1. End-to-end QA Accuracy per question type (Types 1, 2, 3, 4) against practice/answers.jsonl.
2. Building Count Mean Absolute Error (MAE) and MAPE against ground truth.
3. Strict Schema Validation Pass Rate (verifies AnswerResponse and submission format {id, answer, region}).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from pydantic import ValidationError

from sklearn.metrics import f1_score, r2_score

from src.data_utils import DAMAGE_NORMALIZE, parse_wkt_polygon, polygon_bbox
from src.pipeline import DisasterVAPipeline, get_pipeline
from src.qa_parser import parse_question
from src.schema import AnswerResponse, SceneJSON
from src.spatial_engine import answer_question
from src.stage1_dataset import load_label_polygons
from src.stage2_dataset import (
    DAMAGE_LEVELS,
    DAMAGE_TO_IDX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    extract_padded_crop,
)


def evaluate_stage2_damage_on_labels(
    pipeline: DisasterVAPipeline,
    practice_dir: Path,
    pair_ids: List[str],
) -> Dict[str, Any]:
    """Evaluate Stage 2 Siamese classifier on ground-truth building crops across practice pairs."""
    labels_dir = practice_dir / "labels"
    images_dir = practice_dir / "images"

    y_true = []
    y_pred = []

    for pid in pair_ids:
        label_file = labels_dir / f"{pid}.json"
        pre_file = images_dir / f"{pid}_pre.jpg"
        post_file = images_dir / f"{pid}_post.jpg"
        if not label_file.exists() or not pre_file.exists() or not post_file.exists():
            continue

        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        pre_img = np.array(Image.open(pre_file).convert("RGB"), dtype=np.uint8)
        post_img = np.array(Image.open(post_file).convert("RGB"), dtype=np.uint8)

        crops_pre = []
        crops_post = []
        pair_labels = []

        for feat in data.get("features", {}).get("xy", []):
            raw_subtype = feat.get("properties", {}).get("subtype", "no-damage")
            if raw_subtype == "un-classified":
                continue
            canonical = DAMAGE_NORMALIZE.get(raw_subtype, "no-damage")
            if canonical not in DAMAGE_TO_IDX:
                continue

            wkt = feat.get("wkt", "")
            if not wkt:
                continue
            try:
                poly = parse_wkt_polygon(wkt)
                if len(poly) < 3:
                    continue
                bbox = polygon_bbox(poly)
            except Exception:
                continue

            c_pre = extract_padded_crop(pre_img, bbox, pad_factor=1.3, crop_size=96, jitter=False)
            c_post = extract_padded_crop(post_img, bbox, pad_factor=1.3, crop_size=96, jitter=False)

            p_norm = (c_pre.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            q_norm = (c_post.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

            crops_pre.append(p_norm.transpose(2, 0, 1))
            crops_post.append(q_norm.transpose(2, 0, 1))
            pair_labels.append(DAMAGE_TO_IDX[canonical])

        if crops_pre and pipeline.stage2_model is not None:
            with torch.no_grad():
                b_pre = torch.from_numpy(np.array(crops_pre)).float().to(pipeline.device)
                b_post = torch.from_numpy(np.array(crops_post)).float().to(pipeline.device)
                logits = pipeline.stage2_model(b_pre, b_post)
                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

            y_true.extend(pair_labels)
            y_pred.extend(preds)

    if not y_true:
        return {"macro_f1": 0.0, "per_class_f1": {}, "total_evaluated_crops": 0}

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    class_f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    per_class = {DAMAGE_LEVELS[i]: round(float(class_f1s[i]), 4) for i in range(4)}

    return {
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class,
        "total_evaluated_crops": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def load_practice_data(practice_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load practice questions and gold answers."""
    questions_file = practice_dir / "questions.jsonl"
    answers_file = practice_dir / "answers.jsonl"

    questions = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    gold_answers = {}
    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                gold_answers[obj["id"]] = obj

    return questions, gold_answers


def load_gt_building_counts(labels_dir: Path, pair_ids: List[str]) -> Dict[str, int]:
    """Extract true building counts from label JSON files."""
    counts = {}
    for pid in pair_ids:
        label_p = labels_dir / f"{pid}.json"
        polys = load_label_polygons(label_p)
        counts[pid] = len(polys)
    return counts


def run_evaluation(
    practice_dir: Path = Path("Dataset/CIT_Hackathon_PS04_aerial/practice"),
    splits_file: Path = Path("artifacts/splits.json"),
    stage1_ckpt: Optional[Path] = Path("artifacts/stage1_best.pth"),
    stage2_ckpt: Optional[Path] = Path("artifacts/stage2_best.pth"),
    stage1_config: Optional[Path] = Path("artifacts/stage1_config.json"),
    device: Optional[str] = None,
    max_questions: Optional[int] = None,
    output_json: Optional[Path] = Path("outputs/evaluation_results.json"),
) -> Dict[str, Any]:
    """Execute end-to-end evaluation."""
    print("=" * 64)
    print("  PS04 Aerial Disaster QA: End-to-End Evaluation Harness")
    print("=" * 64)

    # 1. Initialize Pipeline (fresh instance with latest stage1 config)
    t_start = time.time()
    pipeline = DisasterVAPipeline(
        stage1_checkpoint=stage1_ckpt if stage1_ckpt and stage1_ckpt.exists() else None,
        stage2_checkpoint=stage2_ckpt if stage2_ckpt and stage2_ckpt.exists() else None,
        stage1_config=stage1_config if stage1_config and stage1_config.exists() else None,
        device=device,
    )

    # 2. Load Practice Dataset
    questions, gold_answers = load_practice_data(practice_dir)
    if max_questions is not None and max_questions > 0:
        questions = questions[:max_questions]

    print(f"Evaluating {len(questions)} questions on practice set...")

    # Unique pair IDs
    unique_pairs = sorted(list({q["pair_id"] for q in questions}))
    gt_counts = load_gt_building_counts(practice_dir / "labels", unique_pairs)

    # 3. Cache Scene Predictions to avoid re-running Stage 1+2 on identical pairs
    print(f"Precomputing scene predictions for {len(unique_pairs)} distinct image pairs...")
    scene_cache: Dict[str, SceneJSON] = {}
    pred_counts: Dict[str, int] = {}
    schema_passed = 0
    total_schema_checks = 0

    for idx, pid in enumerate(unique_pairs):
        pre_img = practice_dir / f"images/{pid}_pre.jpg"
        post_img = practice_dir / f"images/{pid}_post.jpg"

        scene = pipeline.predict_scene(pre_img, post_img, pair_id=pid)
        scene_cache[pid] = scene
        pred_counts[pid] = len(scene.buildings)

        # Validate SceneJSON schema
        total_schema_checks += 1
        try:
            SceneJSON.model_validate(scene.model_dump())
            schema_passed += 1
        except ValidationError:
            pass

    # 4. Building Count MAE & R2 Metric
    count_errors = []
    count_pct_errors = []
    y_true_counts = []
    y_pred_counts = []
    for pid in unique_pairs:
        gt = gt_counts.get(pid, 0)
        pred = pred_counts.get(pid, 0)
        abs_err = abs(pred - gt)
        count_errors.append(abs_err)
        count_pct_errors.append(abs_err / max(gt, 1))
        y_true_counts.append(gt)
        y_pred_counts.append(pred)

    building_mae = float(np.mean(count_errors))
    building_mape = float(np.mean(count_pct_errors)) * 100.0
    building_r2 = float(r2_score(y_true_counts, y_pred_counts)) if len(unique_pairs) > 1 else 0.0

    # 5. Evaluate Stage 2 Damage Classifier on Ground-Truth Crops
    print("Evaluating Stage 2 damage classification on ground-truth crops...")
    stage2_metrics = evaluate_stage2_damage_on_labels(pipeline, practice_dir, unique_pairs)

    # 6. Answer Questions & Evaluate QA Accuracy (Exact & +-10% Count Tolerance)
    type_total = {1: 0, 2: 0, 3: 0, 4: 0}
    type_correct_exact = {1: 0, 2: 0, 3: 0, 4: 0}
    type_correct_tol = {1: 0, 2: 0, 3: 0, 4: 0}
    results_list = []

    for q in questions:
        qid = q["id"]
        pid = q["pair_id"]
        qtext = q["question"]
        gold = gold_answers[qid]
        g_type = gold["type"]
        g_ans = gold["answer"]

        type_total[g_type] += 1
        scene = scene_cache[pid]

        parsed = parse_question(qtext)
        ans_resp = answer_question(scene, parsed, question_id=qid)

        # Validate AnswerResponse schema
        total_schema_checks += 1
        try:
            AnswerResponse.model_validate(ans_resp.model_dump())
            sub_dict = ans_resp.to_submission()
            assert "id" in sub_dict and "answer" in sub_dict and "region" in sub_dict
            schema_passed += 1
        except Exception:
            pass

        # Exact match
        is_exact = str(ans_resp.answer).strip() == str(g_ans).strip()
        if is_exact:
            type_correct_exact[g_type] += 1

        # Tolerance match (+-10% or +-1 for Type 1 count questions; exact for types 2, 3, 4)
        is_tol = is_exact
        if g_type == 1:
            try:
                pred_v = float(ans_resp.answer)
                gold_v = float(g_ans)
                allowed_delta = max(1.0, 0.10 * gold_v)
                is_tol = abs(pred_v - gold_v) <= allowed_delta
            except (ValueError, TypeError):
                is_tol = is_exact

        if is_tol:
            type_correct_tol[g_type] += 1

        results_list.append({
            "id": qid,
            "pair_id": pid,
            "type": g_type,
            "question": qtext,
            "gold_answer": g_ans,
            "predicted_answer": ans_resp.answer,
            "exact_match": is_exact,
            "tol_match": is_tol,
            "gold_region": gold.get("region"),
            "predicted_region": ans_resp.region,
        })

    total_exact = sum(type_correct_exact.values())
    total_tol = sum(type_correct_tol.values())
    total_q = sum(type_total.values())
    overall_exact_acc = (total_exact / max(total_q, 1)) * 100.0
    overall_tol_acc = (total_tol / max(total_q, 1)) * 100.0
    schema_pass_pct = (schema_passed / max(total_schema_checks, 1)) * 100.0

    elapsed = time.time() - t_start

    # 7. Summary Report
    print("\n" + "=" * 64)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 64)
    print(f"Overall QA Accuracy (Exact):       {total_exact}/{total_q} ({overall_exact_acc:.1f}%)")
    print(f"Overall QA Accuracy (+-10% Tol):   {total_tol}/{total_q} ({overall_tol_acc:.1f}%)")
    print(f"Schema Validation Rate:            {schema_passed}/{total_schema_checks} ({schema_pass_pct:.1f}%)")
    print(f"Building Count MAE:                {building_mae:.2f} buildings/image (MAPE: {building_mape:.1f}%)")
    print(f"Building Count R2 Score:           {building_r2:.4f}")
    print(f"Stage 2 Damage Macro F1:           {stage2_metrics.get('macro_f1', 0.0):.4f}")
    print(f"Stage 2 Per-Class F1:              {stage2_metrics.get('per_class_f1', {})}")
    print(f"Total Evaluation Time:             {elapsed:.1f}s")
    print("-" * 64)
    print("Accuracy Breakdown by Question Type:")
    type_names = {
        1: "Type 1 (Counting)",
        2: "Type 2 (Quadrant Max)",
        3: "Type 3 (Half Comparison)",
        4: "Type 4 (Out-of-Scope / Unknown)",
    }
    for t in (1, 2, 3, 4):
        ce = type_correct_exact[t]
        ct = type_correct_tol[t]
        n = type_total[t]
        p_ex = (ce / max(n, 1)) * 100.0
        p_tl = (ct / max(n, 1)) * 100.0
        print(f"  {type_names[t]:34s}: Exact {ce:2d}/{n:2d} ({p_ex:5.1f}%) | Tol {ct:2d}/{n:2d} ({p_tl:5.1f}%)")
    print("=" * 64)

    results_summary = {
        "overall_exact_accuracy": round(overall_exact_acc, 2),
        "overall_tolerance_accuracy": round(overall_tol_acc, 2),
        "total_questions": total_q,
        "exact_correct_questions": total_exact,
        "tolerance_correct_questions": total_tol,
        "per_type_accuracy": {
            t: {
                "exact_correct": type_correct_exact[t],
                "tolerance_correct": type_correct_tol[t],
                "total": type_total[t],
                "exact_pct": round(100.0 * type_correct_exact[t] / max(type_total[t], 1), 2),
                "tolerance_pct": round(100.0 * type_correct_tol[t] / max(type_total[t], 1), 2),
            }
            for t in (1, 2, 3, 4)
        },
        "stage1_metrics": {
            "building_count_mae": round(building_mae, 2),
            "building_count_mape": round(building_mape, 2),
            "building_count_r2": round(building_r2, 4),
            "y_true_counts": y_true_counts,
            "y_pred_counts": y_pred_counts,
        },
        "stage2_metrics": stage2_metrics,
        "schema_validation_rate": round(schema_pass_pct, 2),
        "elapsed_seconds": round(elapsed, 2),
        "detailed_results": results_list,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2)
        print(f"Detailed evaluation report saved to: {output_json}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="PS04 End-to-End Evaluation")
    parser.add_argument("--practice_dir", type=str, default="Dataset/CIT_Hackathon_PS04_aerial/practice")
    parser.add_argument("--splits_file", type=str, default="artifacts/splits.json")
    parser.add_argument("--stage1_ckpt", type=str, default="artifacts/stage1_best.pth")
    parser.add_argument("--stage2_ckpt", type=str, default="artifacts/stage2_best.pth")
    parser.add_argument("--stage1_config", type=str, default="artifacts/stage1_config.json")
    parser.add_argument("--output_json", type=str, default="outputs/evaluation_results.json")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_evaluation(
        practice_dir=Path(args.practice_dir),
        splits_file=Path(args.splits_file),
        stage1_ckpt=Path(args.stage1_ckpt) if args.stage1_ckpt else None,
        stage2_ckpt=Path(args.stage2_ckpt) if args.stage2_ckpt else None,
        stage1_config=Path(args.stage1_config) if args.stage1_config else None,
        output_json=Path(args.output_json) if args.output_json else None,
        max_questions=args.max_questions,
        device=args.device,
    )


if __name__ == "__main__":
    main()
