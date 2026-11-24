"""
FastAPI Backend Server for PS04 Aerial Disaster Visual Intelligence.

Connects the React frontend to:
  - Stage 1: Building Instance Localization (Unet-ResNet34 + Watershed)
  - Stage 2: Siamese ResNet18 Damage Classifier
  - Stage 3: Deterministic Spatial Geometry Engine
  - Conversational Agent: Grounded Groq LLM + Structured Intent Extraction
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_agent import ask_chat_agent, compute_scene_summary, get_active_groq_model
from src.pipeline import get_pipeline
from src.schema import BuildingInfo, SceneJSON
from src.visualizer import render_damage_overlay, render_evidence_map

# Setup Directories
VIZ_DIR = PROJECT_ROOT / "outputs" / "viz"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AeroPulse PS04 Disaster Intelligence API",
    description="High-throughput aerial damage assessment and grounded VQA backend",
    version="2.0.0",
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
#  Scene Cache Management
# ═══════════════════════════════════════════════════════════════

SCENE_CACHE: Dict[str, SceneJSON] = {}
DEMO_RAW: Dict[str, Any] = {}

cache_file = PROJECT_ROOT / "artifacts" / "demo_cache.json"
if cache_file.exists():
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            DEMO_RAW = json.load(f)
            for pid, item in DEMO_RAW.items():
                bldgs = [BuildingInfo.model_validate(b) for b in item["buildings"]]
                SCENE_CACHE[pid] = SceneJSON(
                    pair_id=item["pair_id"],
                    pre_image_path=item["pre_image"],
                    post_image_path=item["post_image"],
                    image_size=[1024, 1024],
                    disaster=item.get("disaster", ""),
                    disaster_type=item.get("disaster_type", ""),
                    buildings=bldgs,
                    source="ground_truth",
                )
        print(f"Loaded {len(SCENE_CACHE)} pre-cached scenes into memory.")
    except Exception as e:
        print(f"Warning: Failed to load demo cache: {e}")


def get_or_render_overlay(scene: SceneJSON) -> str:
    """Ensure full damage overlay image exists and return relative static URL."""
    out_name = f"{scene.pair_id}_damage.png"
    out_path = VIZ_DIR / out_name
    if not out_path.exists():
        post_path = PROJECT_ROOT / scene.post_image_path
        if post_path.exists():
            render_damage_overlay(
                post_image_path=post_path,
                scene=scene,
                output_path=out_path,
            )
    return f"/api/files/viz/{out_name}"


# ═══════════════════════════════════════════════════════════════
#  API Routes
# ═══════════════════════════════════════════════════════════════

@app.get("/api/status")
def get_system_status():
    """Return live hardware, model checkpoint, and LLM agent status."""
    stage1_ckpt = PROJECT_ROOT / "artifacts" / "stage1_best.pth"
    stage2_ckpt = PROJECT_ROOT / "artifacts" / "stage2_best.pth"

    groq_key = os.environ.get("GROQ_API_KEY", "")
    active_model = get_active_groq_model(groq_key) if groq_key else "Offline (Deterministic)"

    return {
        "status": "online",
        "timestamp": time.time(),
        "stage1_ready": stage1_ckpt.exists(),
        "stage2_ready": stage2_ckpt.exists(),
        "active_llm": f"Groq ({active_model})" if groq_key else "Deterministic Spatial Rules (0ms)",
        "cached_scenes_count": len(SCENE_CACHE),
    }


@app.get("/api/scenes")
def list_scenes():
    """List all available disaster scenes with high-level summaries and preview URLs."""
    results = []
    for pid, scene in SCENE_CACHE.items():
        summary = compute_scene_summary(scene)
        overlay_url = get_or_render_overlay(scene)
        results.append({
            "pair_id": pid,
            "disaster": scene.disaster or "Disaster Incident",
            "disaster_type": scene.disaster_type or "Satellite View",
            "total_buildings": summary["total_buildings"],
            "total_damaged": summary["total_damaged"],
            "damage_breakdown": summary["by_damage_tier"],
            "pre_url": f"/api/files/raw/{scene.pre_image_path}",
            "post_url": f"/api/files/raw/{scene.post_image_path}",
            "overlay_url": overlay_url,
        })
    return {"scenes": results}


@app.get("/api/scenes/{pair_id}")
def get_scene(pair_id: str):
    """Get full details, damage summary, and image URLs for a specific scene."""
    if pair_id not in SCENE_CACHE:
        raise HTTPException(status_code=404, detail=f"Scene '{pair_id}' not found.")
    scene = SCENE_CACHE[pair_id]
    summary = compute_scene_summary(scene)
    overlay_url = get_or_render_overlay(scene)

    return {
        "pair_id": scene.pair_id,
        "disaster": scene.disaster,
        "disaster_type": scene.disaster_type,
        "summary": summary,
        "buildings_count": len(scene.buildings),
        "pre_url": f"/api/files/raw/{scene.pre_image_path}",
        "post_url": f"/api/files/raw/{scene.post_image_path}",
        "overlay_url": overlay_url,
    }


@app.post("/api/upload")
async def upload_pair(
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
    pair_id: Optional[str] = Form(None),
):
    """Upload custom satellite image pair, run live neural inference, and cache scene."""
    pid = pair_id or f"scene_{int(time.time())}"
    pre_filename = f"{pid}_pre.jpg"
    post_filename = f"{pid}_post.jpg"

    pre_save_path = UPLOADS_DIR / pre_filename
    post_save_path = UPLOADS_DIR / post_filename

    with open(pre_save_path, "wb") as f_pre:
        shutil.copyfileobj(pre_image.file, f_pre)
    with open(post_save_path, "wb") as f_post:
        shutil.copyfileobj(post_image.file, f_post)

    # Run Live Neural Pipeline (Stage 1 + Stage 2)
    t0 = time.time()
    pipeline = get_pipeline()
    scene = pipeline.predict_scene(
        pre_path=pre_save_path,
        post_path=post_save_path,
        pair_id=pid,
    )
    SCENE_CACHE[pid] = scene
    summary = compute_scene_summary(scene)
    overlay_url = get_or_render_overlay(scene)
    exec_time = round(time.time() - t0, 3)

    return {
        "pair_id": pid,
        "exec_time_seconds": exec_time,
        "summary": summary,
        "pre_url": f"/api/files/upload/{pre_filename}",
        "post_url": f"/api/files/upload/{post_filename}",
        "overlay_url": overlay_url,
    }


class ChatRequest(BaseModel):
    pair_id: str
    question: str
    question_id: Optional[str] = None


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """Process a natural language disaster VQA query over a specific scene."""
    if req.pair_id not in SCENE_CACHE:
        raise HTTPException(status_code=404, detail=f"Scene '{req.pair_id}' not found.")

    scene = SCENE_CACHE[req.pair_id]
    q_id = req.question_id or f"chat_{int(time.time() * 1000)}"

    t0 = time.time()
    res = ask_chat_agent(question=req.question, scene=scene, question_id=q_id)
    exec_dur = round(time.time() - t0, 3)

    ans_resp = res["answer_response"]
    evidence_url = None

    # Render highlighted visual evidence map if question has spatial target
    if ans_resp.region and ans_resp.region != "full":
        ev_filename = f"{req.pair_id}_ev_{int(time.time() * 1000)}.png"
        ev_path = VIZ_DIR / ev_filename
        post_img_path = PROJECT_ROOT / scene.post_image_path
        if post_img_path.exists():
            try:
                render_evidence_map(
                    post_image_path=post_img_path,
                    scene=scene,
                    answer=ans_resp,
                    output_path=ev_path,
                )
                evidence_url = f"/api/files/viz/{ev_filename}"
            except Exception as e:
                print(f"Warning: Failed to render evidence overlay: {e}")

    return {
        "reply": res["natural_response"],
        "natural_response": res["natural_response"],
        "submission_json": res["submission_json"],
        "full_json": res["full_json"],
        "backend": res["backend"],
        "region": res["region"],
        "supporting_count": len(res.get("supporting_building_ids", [])),
        "evidence_url": evidence_url,
        "scene_summary": res.get("scene_summary", {}),
        "exec_time_seconds": exec_dur,
    }


# ═══════════════════════════════════════════════════════════════
#  Static File Delivery Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/api/files/raw/{file_path:path}")
def serve_raw_image(file_path: str):
    """Serve imagery from dataset directory."""
    full_path = PROJECT_ROOT / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {file_path}")
    return FileResponse(full_path)


@app.get("/api/files/viz/{filename}")
def serve_viz_image(filename: str):
    """Serve generated visual overlays."""
    full_path = VIZ_DIR / filename
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Overlay not found: {filename}")
    return FileResponse(full_path)


@app.get("/api/files/upload/{filename}")
def serve_uploaded_image(filename: str):
    """Serve user-uploaded imagery."""
    full_path = UPLOADS_DIR / filename
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Uploaded image not found: {filename}")
    return FileResponse(full_path)


# Mount Production Frontend Dist if built
DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
