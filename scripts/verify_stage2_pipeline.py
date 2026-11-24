"""
Verification suite for Stage 2 (Siamese Damage Classifier) and Unified Pipeline:
1. Test Stage2SiameseCropDataset with 1.3x padded 96x96 crops, jitter, and WeightedRandomSampler.
2. Test SiameseDamageClassifier forward pass [f_pre, f_post, f_post - f_pre] -> 4 damage classes.
3. Test unified pipeline end-to-end: predict_scene() and answer_disaster_question() with visual rendering.
"""

import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_dataset import (
    Stage2SiameseCropDataset,
    get_stage2_dataloaders,
    DAMAGE_LEVELS,
    DAMAGE_TO_IDX,
)
from train_stage2 import SiameseDamageClassifier
from src.pipeline import predict_scene, answer_disaster_question, DisasterVAPipeline
from src.schema import AnswerResponse, SceneJSON


def test_stage2_dataset():
    print("\n[1/3] Testing Stage2SiameseCropDataset (1.3x Padded 96x96 Crops & Jitter)...")
    train_ds, val_ds, train_loader, val_loader = get_stage2_dataloaders(
        splits_path="artifacts/splits.json",
        data_root="Dataset/CIT_Hackathon_PS04_aerial/train",
        batch_size=4,
        val_batch_size=4,
        crop_size=96,
        pad_factor=1.3,
        use_weighted_sampler=True,
    )

    print(f"  Indexed {len(train_ds)} train building crops, {len(val_ds)} val crops.")
    print(f"  Train class distribution: {train_ds.get_class_counts()}")

    # Fetch 1 batch from train_loader
    for pre, post, labels, uids in train_loader:
        assert pre.shape == (4, 3, 96, 96), f"Unexpected pre shape: {pre.shape}"
        assert post.shape == (4, 3, 96, 96), f"Unexpected post shape: {post.shape}"
        assert labels.shape == (4,), f"Unexpected labels shape: {labels.shape}"
        assert all(0 <= l < 4 for l in labels)
        print(f"  ✓ Train batch shapes verified: pre={pre.shape}, post={post.shape}, labels={labels.tolist()}, uids={uids}")
        break


def test_siamese_model():
    print("\n[2/3] Testing SiameseDamageClassifier ([f_pre, f_post, f_post - f_pre] -> 4 classes)...")
    model = SiameseDamageClassifier(pretrained=True, num_classes=4, dropout=0.3)
    model.eval()

    dummy_pre = torch.randn(4, 3, 96, 96)
    dummy_post = torch.randn(4, 3, 96, 96)
    dummy_labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    with torch.no_grad():
        logits = model(dummy_pre, dummy_post)

    assert logits.shape == (4, 4), f"Unexpected logits shape: {logits.shape}"

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    loss = criterion(logits, dummy_labels)
    print(f"  ✓ Siamese forward pass OK. Output shape: {logits.shape}, Sample CE loss: {loss.item():.4f}")


def test_unified_pipeline():
    print("\n[3/3] Testing Unified Pipeline (predict_scene + answer_disaster_question)...")
    pre_path = "Dataset/CIT_Hackathon_PS04_aerial/practice/images/p11b4ad02_pre.jpg"
    post_path = "Dataset/CIT_Hackathon_PS04_aerial/practice/images/p11b4ad02_post.jpg"
    question = "How many undamaged buildings are there in the top-right quadrant?"

    t0 = time.time()
    # 1. Test predict_scene
    scene = predict_scene(
        pre_path=pre_path,
        post_path=post_path,
        pair_id="p11b4ad02",
        device="cpu",
    )
    t_scene = time.time() - t0

    assert isinstance(scene, SceneJSON)
    assert scene.pair_id == "p11b4ad02"
    print(f"  ✓ predict_scene() produced SceneJSON in {t_scene:.2f}s with {len(scene.buildings)} classified buildings.")
    if scene.buildings:
        b0 = scene.buildings[0]
        print(f"    Sample building: id={b0.id}, damage={b0.damage}, conf={b0.confidence}, center={b0.center}")

    # 2. Test answer_disaster_question with visualization
    t1 = time.time()
    ans = answer_disaster_question(
        pre_img=pre_path,
        post_img=post_path,
        question=question,
        qid="q_test_01",
        pair_id="p11b4ad02",
        render_viz=True,
        output_dir="outputs",
        device="cpu",
    )
    t_ans = time.time() - t1

    assert isinstance(ans, AnswerResponse)
    assert ans.id == "q_test_01"
    assert ans.question_type == 1
    print(f"  ✓ answer_disaster_question() returned: answer={ans.answer!r}, region={ans.region} in {t_ans:.2f}s.")

    # Check generated visualization files
    overlay_path = Path("outputs/p11b4ad02_damage_overlay.png")
    evidence_path = Path("outputs/p11b4ad02_evidence_map.png")
    assert overlay_path.exists(), "damage overlay was not saved"
    assert evidence_path.exists(), "evidence map was not saved"
    print(f"  ✓ Visual evidence saved to {overlay_path} and {evidence_path}.")


if __name__ == "__main__":
    print("=" * 64)
    print("  PS04 Stage 2 & Pipeline Verification Suite")
    print("=" * 64)
    test_stage2_dataset()
    test_siamese_model()
    test_unified_pipeline()
    print("\n" + "=" * 64)
    print("  ✅ ALL STAGE-2 & PIPELINE CHECKS PASSED")
    print("=" * 64)
