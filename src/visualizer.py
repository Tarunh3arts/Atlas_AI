"""
Visual evidence renderer for PS04 Aerial View Image Reasoning.

Produces two kinds of visualisation:

1. **Damage overlay** — colour-coded building polygons on the post-disaster
   image (green / yellow / orange / red).
2. **Evidence map** — dims non-relevant regions and outlines the exact
   buildings counted for a specific question.

Requires ``Pillow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .schema import (
    AnswerResponse,
    BuildingInfo,
    HALVES,
    QUADRANTS,
    SceneJSON,
)


# ═══════════════════════════════════════════════════════════════
#  Colour palette
# ═══════════════════════════════════════════════════════════════

DAMAGE_COLORS: dict[str, tuple[int, int, int]] = {
    "no-damage":    (0,   200, 0),    # Green
    "minor-damage": (255, 255, 0),    # Yellow
    "major-damage": (255, 136, 0),    # Orange
    "destroyed":    (255, 0,   0),    # Red
}

_OUTLINE_COLOR = (0, 255, 255)   # Cyan outline for evidence buildings
_DIM_COLOR     = (0, 0, 0)       # Black overlay for dimmed regions
_DIM_ALPHA     = 120             # Opacity of the dim layer (0–255)
_FILL_ALPHA    = 90              # Opacity of damage fill polygons
_EVIDENCE_FILL_ALPHA = 150       # Brighter fill for evidence buildings


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _poly_tuples(polygon: list[list[float]]) -> list[tuple[float, float]]:
    """Convert [[x, y], …] → [(x, y), …] for PIL."""
    return [(p[0], p[1]) for p in polygon]


def _try_load_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font; fall back to default bitmap."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


# ═══════════════════════════════════════════════════════════════
#  1.  Damage overlay
# ═══════════════════════════════════════════════════════════════

def render_damage_overlay(
    post_image_path: str | Path | None = None,
    scene: Optional[SceneJSON] = None,
    output_path: str | Path = "outputs/damage_overlay.png",
    *,
    fill_alpha: int = _FILL_ALPHA,
    outline_width: int = 2,
    draw_labels: bool = False,
) -> Path:
    """Draw colour-coded damage polygons on the post-disaster image.

    Parameters
    ----------
    post_image_path
        Path to the post-disaster JPEG. If None, resolves from scene.post_image_path.
    scene
        The unified scene (GT or predicted).
    output_path
        Where to save the rendered PNG.
    fill_alpha
        Transparency of filled polygons (0 = invisible, 255 = opaque).
    outline_width
        Width of polygon outlines.
    draw_labels
        If ``True``, write the building UID + damage class next to each building.

    Returns
    -------
    Path
        The ``output_path``, for convenience.
    """
    if post_image_path is None and scene is not None:
        post_image_path = scene.post_image_path
    if not post_image_path:
        raise ValueError("post_image_path must be provided or present in scene.post_image_path")

    post_path_obj = Path(post_image_path)
    # Ensure post-disaster image is targeted (safeguard against accidental _pre path passing)
    if "_pre." in post_path_obj.name:
        candidate_post = post_path_obj.parent / post_path_obj.name.replace("_pre.", "_post.")
        if candidate_post.exists():
            post_path_obj = candidate_post
    elif not post_path_obj.exists() and scene is not None and Path(scene.post_image_path).exists():
        post_path_obj = Path(scene.post_image_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = Image.open(post_path_obj).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if scene is not None:
        for b in scene.buildings:
            color = DAMAGE_COLORS.get(b.damage, (128, 128, 128))
            fill = (*color, fill_alpha)
            outline = (*color, 255)
            pts = _poly_tuples(b.polygon)
            draw.polygon(pts, fill=fill, outline=outline, width=outline_width)

    composite = Image.alpha_composite(base, overlay)

    if draw_labels and scene is not None:
        label_draw = ImageDraw.Draw(composite)
        font = _try_load_font(11)
        for b in scene.buildings:
            label_draw.text(
                (b.center[0] + 4, b.center[1] - 6),
                f"{b.id} {b.damage}",
                fill=(255, 255, 255, 220),
                font=font,
            )

    composite.convert("RGB").save(output_path, quality=92)
    return output_path


# ═══════════════════════════════════════════════════════════════
#  2.  Evidence map
# ═══════════════════════════════════════════════════════════════

def render_evidence_map(
    post_image_path: str | Path | None = None,
    scene: Optional[SceneJSON] = None,
    answer: Optional[AnswerResponse] = None,
    output_path: str | Path = "outputs/evidence_map.png",
    *,
    highlight_building_ids: Optional[Sequence[str]] = None,
    dim_alpha: int = _DIM_ALPHA,
    evidence_outline_width: int = 3,
) -> Path:
    """Render an evidence map that highlights the relevant area and buildings on the post-disaster image.

    - Non-relevant quadrants/halves are dimmed.
    - Buildings that contribute to the answer are outlined in cyan with
      a brighter fill.
    - Other buildings get a subtle damage-coloured fill.

    Parameters
    ----------
    post_image_path
        Path to the post-disaster JPEG. If None, resolves from scene.post_image_path.
    scene
        The unified scene.
    answer
        The ``AnswerResponse`` for the question being visualised.
    output_path
        Where to save the rendered PNG.
    highlight_building_ids
        Building IDs to highlight.  Defaults to
        ``answer.supporting_building_ids``.
    dim_alpha
        Opacity of the dark overlay on non-relevant regions.
    evidence_outline_width
        Outline width for highlighted buildings.

    Returns
    -------
    Path
        The ``output_path``.
    """
    if post_image_path is None and scene is not None:
        post_image_path = scene.post_image_path
    if not post_image_path:
        raise ValueError("post_image_path must be provided or present in scene.post_image_path")

    post_path_obj = Path(post_image_path)
    # Ensure post-disaster image is targeted (safeguard against accidental _pre path passing)
    if "_pre." in post_path_obj.name:
        candidate_post = post_path_obj.parent / post_path_obj.name.replace("_pre.", "_post.")
        if candidate_post.exists():
            post_path_obj = candidate_post
    elif not post_path_obj.exists() and scene is not None and Path(scene.post_image_path).exists():
        post_path_obj = Path(scene.post_image_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if highlight_building_ids is None and answer is not None:
        highlight_building_ids = answer.supporting_building_ids
    highlight_set = set(highlight_building_ids or [])

    base = Image.open(post_path_obj).convert("RGBA")
    w, h = base.size

    # ── Dim layer: darken everything outside the answer region ──
    dim_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dim_draw  = ImageDraw.Draw(dim_layer)

    if answer is not None and answer.region is not None:
        # Fill entire image with dim, then cut out the relevant region
        dim_draw.rectangle([(0, 0), (w, h)], fill=(*_DIM_COLOR, dim_alpha))
        rx1, ry1, rx2, ry2 = answer.region
        dim_draw.rectangle([(rx1, ry1), (rx2, ry2)], fill=(0, 0, 0, 0))

    base = Image.alpha_composite(base, dim_layer)

    # ── Building overlay ────────────────────────────────────────
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    if scene is not None:
        for b in scene.buildings:
            pts = _poly_tuples(b.polygon)
            color = DAMAGE_COLORS.get(b.damage, (128, 128, 128))

            if b.id in highlight_set:
                # Highlighted: brighter fill + cyan outline
                fill = (*color, _EVIDENCE_FILL_ALPHA)
                draw.polygon(pts, fill=fill, outline=(*_OUTLINE_COLOR, 255),
                             width=evidence_outline_width)
            else:
                # Non-highlighted: subtle fill, thin outline
                fill = (*color, 40)
                draw.polygon(pts, fill=fill, outline=(*color, 100), width=1)

    composite = Image.alpha_composite(base, overlay)

    # ── Question / answer text banner ───────────────────────────
    if answer is not None:
        final_draw = ImageDraw.Draw(composite)
        font = _try_load_font(16)
        q_text = str(answer.question or "")[:90]
        a_text = f"Answer: {answer.answer}"
        # Semi-transparent banner at the top
        banner_h = 52
        banner = Image.new("RGBA", (w, banner_h), (0, 0, 0, 180))
        composite.paste(banner, (0, 0), banner)
        final_draw.text((8, 4),  q_text, fill=(255, 255, 255, 240), font=font)
        final_draw.text((8, 26), a_text, fill=(100, 255, 100, 240), font=font)

    composite.convert("RGB").save(output_path, quality=92)
    return output_path


# ═══════════════════════════════════════════════════════════════
#  Convenience: both overlays in one call
# ═══════════════════════════════════════════════════════════════

def render_all(
    post_image_path: str | Path,
    scene: SceneJSON,
    answer: AnswerResponse,
    output_dir: str | Path,
    *,
    prefix: str = "",
) -> dict[str, Path]:
    """Render both the damage overlay and the evidence map.

    Returns a dict with keys ``"damage_overlay"`` and ``"evidence_map"``.
    """
    output_dir = Path(output_dir)
    pfx = f"{prefix}_" if prefix else ""
    return {
        "damage_overlay": render_damage_overlay(
            post_image_path, scene,
            output_dir / f"{pfx}damage_overlay.png",
        ),
        "evidence_map": render_evidence_map(
            post_image_path, scene, answer,
            output_dir / f"{pfx}evidence_map.png",
        ),
    }
