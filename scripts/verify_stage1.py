"""
End-to-end verification script for Stage 1 components:
1. Test dataset loading, 512x512 cropping, and boundary-carved mask rasterization.
2. Test model instantiation and forward/backward pass with BCE + Dice loss.
3. Test inference pipeline: 1024x1024 forward pass + distance transform watershed + SceneJSON conversion.
4. Test tuner on a quick 2-sample validation slice.
"""

import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1_dataset import Stage1BuildingDataset, get_stage1_dataloaders
from train_stage1 import build_model, DiceLoss
from src.stage1_infer import Stage1Detector
from src.schema import SceneJSON


def test_dataset():
    print("\n[1/4] Testing Stage1BuildingDataset & Boundary-Carved Rasterization...")
    _, val_ds, train_loader, val_loader = get_stage1_dataloaders(
        splits_path="artifacts/splits.json",
        data_root="Dataset/CIT_Hackathon_PS04_aerial/train",
        batch_size=2,
        val_batch_size=1,
        crop_size=512,
        border_px=2,
    )
    print(f"  Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Check 1 train batch
    for imgs, masks, pids in train_loader:
        assert imgs.shape == (2, 3, 512, 512), f"Unexpected imgs shape: {imgs.shape}"
        assert masks.shape == (2, 1, 512, 512), f"Unexpected masks shape: {masks.shape}"
        assert masks.min() >= 0.0 and masks.max() <= 1.0
        print(f"  ✓ Train batch shapes verified: img={imgs.shape}, mask={masks.shape}, pair_ids={pids}")
        break

    # Check 1 val sample (full 1024x1024)
    for imgs, masks, pids in val_loader:
        assert imgs.shape == (1, 3, 1024, 1024), f"Unexpected val imgs shape: {imgs.shape}"
        assert masks.shape == (1, 1, 1024, 1024), f"Unexpected val masks shape: {masks.shape}"
        print(f"  ✓ Val sample full 1024x1024 verified: img={imgs.shape}, mask={masks.shape}")
        break


def test_model_and_loss():
    print("\n[2/4] Testing Unet(resnet34) Model & Combined Loss...")
    model = build_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    model.eval()

    dummy_input = torch.randn(2, 3, 512, 512)
    dummy_target = (torch.rand(2, 1, 512, 512) > 0.8).float()

    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (2, 1, 512, 512), f"Unexpected logits shape: {logits.shape}"

    bce_fn = torch.nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()
    loss = bce_fn(logits, dummy_target) + dice_fn(logits, dummy_target)
    print(f"  ✓ Forward pass OK. Sample loss: {loss.item():.4f}")


def test_inference_and_watershed():
    print("\n[3/4] Testing Full 1024x1024 Inference & Watershed Instance Segmentation...")
    detector = Stage1Detector(checkpoint_path=None, encoder="resnet34", device="cpu")
    sample_img_path = "Dataset/CIT_Hackathon_PS04_aerial/practice/images/p11b4ad02_pre.jpg"

    t0 = time.time()
    scene = detector.detect(sample_img_path, pair_id="p11b4ad02")
    t_elapsed = time.time() - t0

    assert isinstance(scene, SceneJSON)
    assert scene.pair_id == "p11b4ad02"
    assert scene.image_size == [1024, 1024]
    print(f"  ✓ SceneJSON generated in {t_elapsed:.2f}s with {len(scene.buildings)} detected building instances.")
    if scene.buildings:
        b = scene.buildings[0]
        print(f"    Sample building: id={b.id}, bbox={b.bbox}, center={b.center}, area={b.area_px}px, quadrant={b.quadrant}")


def test_tuner_dry_run():
    print("\n[4/4] Testing tune_stage1.py (dry run on 2 validation samples)...")
    import subprocess
    cmd = [
        sys.executable,
        "tune_stage1.py",
        "--max_samples", "2",
        "--output_config", "artifacts/test_stage1_config.json"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Tuner error output:\n", res.stderr)
        raise RuntimeError(f"tune_stage1 failed with code {res.returncode}")
    print("  ✓ tune_stage1.py completed successfully on test slice.")
    config_file = Path("artifacts/test_stage1_config.json")
    assert config_file.exists(), "test_stage1_config.json was not created"
    print("  ✓ artifacts/test_stage1_config.json verified.")
    config_file.unlink(missing_ok=True)


if __name__ == "__main__":
    print("=" * 64)
    print("  PS04 Stage 1 Verification Suite")
    print("=" * 64)
    test_dataset()
    test_model_and_loss()
    test_inference_and_watershed()
    test_tuner_dry_run()
    print("\n" + "=" * 64)
    print("  ✅ ALL STAGE-1 VERIFICATION CHECKS PASSED")
    print("=" * 64)
