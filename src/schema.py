"""
Unified Scene JSON contract and QA answer-response schemas.

Every module downstream of Stage-1 segmentation + Stage-2 damage classification
consumes or produces the dataclasses defined here.  Ground-truth labels and
model predictions share the **same** SceneJSON shape so that the QA engine,
visualisation code, and evaluation harness are source-agnostic.

Spatial constants are taken verbatim from
``Dataset/CIT_Hackathon_PS04_aerial/question_templates.json``.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


# ── Damage tiers (the 4 normalized classes) ─────────────────────
class DamageLevel(str, Enum):
    """Canonical damage tiers used throughout the pipeline."""
    NO_DAMAGE    = "no-damage"
    MINOR_DAMAGE = "minor-damage"
    MAJOR_DAMAGE = "major-damage"
    DESTROYED    = "destroyed"


DAMAGE_LEVELS: list[str] = [d.value for d in DamageLevel]
"""Ordered list: index 0 = no-damage … 3 = destroyed."""

# Map raw xBD subtypes (including un-classified) → canonical 4 tiers
DAMAGE_NORMALIZE: dict[str, str] = {
    "no-damage":    DamageLevel.NO_DAMAGE.value,
    "minor-damage": DamageLevel.MINOR_DAMAGE.value,
    "major-damage": DamageLevel.MAJOR_DAMAGE.value,
    "destroyed":    DamageLevel.DESTROYED.value,
    "un-classified": DamageLevel.NO_DAMAGE.value,   # safest default for counting
}


# ── Spatial constants from question_templates.json ──────────────
IMAGE_WIDTH  = 1024
IMAGE_HEIGHT = 1024
IMAGE_SIZE   = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_MID_X  = IMAGE_WIDTH  // 2   # 512
IMAGE_MID_Y  = IMAGE_HEIGHT // 2   # 512

QUADRANTS: dict[str, list[int]] = {
    "top-left":     [0,             0,              IMAGE_MID_X, IMAGE_MID_Y],
    "top-right":    [IMAGE_MID_X,   0,              IMAGE_WIDTH, IMAGE_MID_Y],
    "bottom-left":  [0,             IMAGE_MID_Y,    IMAGE_MID_X, IMAGE_HEIGHT],
    "bottom-right": [IMAGE_MID_X,   IMAGE_MID_Y,    IMAGE_WIDTH, IMAGE_HEIGHT],
}

HALVES: dict[str, list[int]] = {
    "left":   [0,           0,           IMAGE_MID_X, IMAGE_HEIGHT],
    "right":  [IMAGE_MID_X, 0,           IMAGE_WIDTH, IMAGE_HEIGHT],
    "top":    [0,           0,           IMAGE_WIDTH, IMAGE_MID_Y],
    "bottom": [0,           IMAGE_MID_Y, IMAGE_WIDTH, IMAGE_HEIGHT],
}


# ── Question types ──────────────────────────────────────────────
class QuestionType(int, Enum):
    """The four question types defined in the problem statement."""
    COUNT        = 1   # "How many {level} are there {scope}?"
    QUADRANT_MAX = 2   # "Which quadrant has the most damaged/destroyed buildings?"
    HALF_COMPARE = 3   # "Is the {a} half more damaged than the {b} half?"
    OUT_OF_SCOPE = 4   # Unanswerable – always "cannot be determined"


# ═══════════════════════════════════════════════════════════════
#  Building – single instance
# ═══════════════════════════════════════════════════════════════
class BuildingInfo(BaseModel):
    """One building instance in the unified scene representation.

    Identical schema whether the source is ground-truth labels or model
    predictions; only ``confidence`` and ``id`` prefix differ.
    """
    id: str = Field(
        ...,
        description=(
            "Unique building identifier.  Ground-truth UIDs are 'b0000', "
            "'b0001', …; predicted detections use 'det_0', 'det_1', …"
        ),
    )
    polygon: list[list[float]] = Field(
        ...,
        description="Closed polygon as [[x, y], …] in pixel coordinates.",
    )
    bbox: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Axis-aligned bounding box [x_min, y_min, x_max, y_max].",
    )
    center: list[float] = Field(
        ..., min_length=2, max_length=2,
        description="Centroid [cx, cy] – used for quadrant/half assignment.",
    )
    area_px: float = Field(
        ..., ge=0,
        description="Polygon area in px² (shoelace formula).",
    )
    quadrant: str = Field(
        ...,
        description="Quadrant containing the centroid: top-left | top-right | bottom-left | bottom-right.",
    )
    halves: list[str] = Field(
        ..., min_length=2, max_length=2,
        description=(
            "The two halves the centroid falls in: "
            "[horizontal_half, vertical_half], e.g. ['top', 'left']."
        ),
    )
    damage: str = Field(
        ...,
        description="Normalized damage class: no-damage | minor-damage | major-damage | destroyed.",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="1.0 for ground-truth; model softmax/sigmoid confidence for predictions.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "b0000",
                    "polygon": [
                        [824.0, 307.57], [818.3, 321.84],
                        [817.79, 337.56], [848.32, 345.79],
                        [855.45, 314.7], [824.0, 307.57],
                    ],
                    "bbox": [817.79, 307.57, 855.45, 345.79],
                    "center": [832.77, 325.49],
                    "area_px": 1038.6,
                    "quadrant": "top-right",
                    "halves": ["top", "right"],
                    "damage": "no-damage",
                    "confidence": 1.0,
                }
            ]
        }
    }


# ═══════════════════════════════════════════════════════════════
#  SceneJSON – one processed image pair
# ═══════════════════════════════════════════════════════════════
class SceneJSON(BaseModel):
    """Unified representation of a single pre/post image pair with
    all detected (or ground-truth) buildings and their attributes."""

    pair_id: str = Field(..., description="Unique pair identifier, e.g. 'tcba8c513'.")
    pre_image_path: str = Field(..., description="Relative or absolute path to the pre-disaster image.")
    post_image_path: str = Field(..., description="Relative or absolute path to the post-disaster image.")
    image_size: list[int] = Field(
        default=[IMAGE_WIDTH, IMAGE_HEIGHT],
        min_length=2, max_length=2,
        description="[width, height] in pixels.",
    )
    disaster: str = Field(default="", description="Disaster event name, e.g. 'palu-tsunami'.")
    disaster_type: str = Field(default="", description="Disaster category, e.g. 'tsunami'.")
    buildings: list[BuildingInfo] = Field(default_factory=list)
    source: str = Field(
        default="ground_truth",
        description="'ground_truth' or 'prediction' – provenance tag.",
    )


# ═══════════════════════════════════════════════════════════════
#  AnswerResponse – full QA output (superset of submission format)
# ═══════════════════════════════════════════════════════════════
class AnswerResponse(BaseModel):
    """Complete QA answer with provenance and supporting evidence.

    The submission endpoint only requires ``{id, answer, region}``;
    call :meth:`to_submission` to produce that minimal dict.
    """
    id: str = Field(..., description="Question ID, e.g. 'q0df0717'.")
    pair_id: str = Field(..., description="Image-pair ID this question refers to.")
    question_type: int = Field(..., ge=1, le=4, description="Question type (1–4).")
    question: str = Field(..., description="Original question text.")
    answer: Union[int, str] = Field(
        ...,
        description=(
            "Answer value: int for counts (type 1), quadrant name (type 2), "
            "'yes'/'no' (type 3), or 'cannot be determined' (type 4 / zero buildings)."
        ),
    )
    region: Optional[list[int]] = Field(
        default=None,
        description="Highlighted region [x1, y1, x2, y2] or null.",
    )
    region_name: Optional[str] = Field(
        default=None,
        description="Human-readable region name, e.g. 'left half', 'top-right', or null.",
    )
    supporting_building_ids: list[str] = Field(
        default_factory=list,
        description="UIDs of buildings that contribute to / support the answer.",
    )
    answerable: bool = Field(
        default=True,
        description="False for type-4 (out-of-scope) or zero-building edge cases.",
    )

    # ── Serialisation helpers ───────────────────────────────────
    def to_submission(self) -> dict:
        """Return the minimal ``{id, answer, region}`` dict for submission."""
        return {
            "id": self.id,
            "answer": self.answer,
            "region": self.region,
        }

    def to_full_dict(self) -> dict:
        """Return all fields as a plain dict (matches practice/answers.jsonl + extras)."""
        return self.model_dump()
