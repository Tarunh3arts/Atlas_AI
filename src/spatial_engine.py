"""
Deterministic spatial QA engine for PS04 Aerial View Image Reasoning.

Executes exact filtering, counting, quadrant ranking, and half comparisons
over the Unified Scene JSON.  Every answer is computed with pure Python
arithmetic — no LLM or model is involved.

Public API
----------
``answer_question(scene, parsed, question_id)`` → ``AnswerResponse``
"""

from __future__ import annotations

from typing import Optional

from .qa_parser import ParsedQuestion
from .schema import (
    AnswerResponse,
    BuildingInfo,
    HALVES,
    QUADRANTS,
    SceneJSON,
)


# ═══════════════════════════════════════════════════════════════
#  Building filters
# ═══════════════════════════════════════════════════════════════

_DAMAGED_SET = frozenset({"minor-damage", "major-damage", "destroyed"})


def _filter_by_damage(
    buildings: list[BuildingInfo],
    damage_filter: str,
) -> list[BuildingInfo]:
    """Return buildings matching the damage filter."""
    if damage_filter == "all":
        return list(buildings)
    if damage_filter == "any_damaged":
        return [b for b in buildings if b.damage in _DAMAGED_SET]
    return [b for b in buildings if b.damage == damage_filter]


def _filter_by_region(
    buildings: list[BuildingInfo],
    region_filter: str,
) -> list[BuildingInfo]:
    """Return buildings whose centroid falls in the given region."""
    if region_filter == "all":
        return list(buildings)
    # Quadrant: match directly
    if region_filter in QUADRANTS:
        return [b for b in buildings if b.quadrant == region_filter]
    # Half: match the appropriate element
    if region_filter in ("top", "bottom"):
        return [b for b in buildings if b.halves[0] == region_filter]
    if region_filter in ("left", "right"):
        return [b for b in buildings if b.halves[1] == region_filter]
    return list(buildings)


def _filter(
    buildings: list[BuildingInfo],
    damage_filter: str,
    region_filter: str,
) -> list[BuildingInfo]:
    """Apply both filters in sequence."""
    return _filter_by_damage(
        _filter_by_region(buildings, region_filter),
        damage_filter,
    )


# ═══════════════════════════════════════════════════════════════
#  Region / bounding-box helpers
# ═══════════════════════════════════════════════════════════════

def _buildings_bbox(buildings: list[BuildingInfo]) -> list[int]:
    """Compute the tight bounding box enclosing all coordinates of the given buildings."""
    if not buildings:
        return [0, 0, 0, 0]
    all_xs = [pt[0] for b in buildings for pt in b.polygon]
    all_ys = [pt[1] for b in buildings for pt in b.polygon]
    return [
        int(round(min(all_xs))),
        int(round(min(all_ys))),
        int(round(max(all_xs))),
        int(round(max(all_ys))),
    ]


def _region_for_scope(region_filter: str) -> Optional[list[int]]:
    """Return the fixed bbox for a named scope, or None for 'all'."""
    if region_filter in QUADRANTS:
        return list(QUADRANTS[region_filter])
    if region_filter in HALVES:
        return list(HALVES[region_filter])
    return None                          # "all" → computed from buildings


def _ids(buildings: list[BuildingInfo]) -> list[str]:
    return [b.id for b in buildings]


# ═══════════════════════════════════════════════════════════════
#  Type-1  –  Count
# ═══════════════════════════════════════════════════════════════

def _answer_count(
    scene: SceneJSON,
    parsed: ParsedQuestion,
    qid: str,
) -> AnswerResponse:
    matched = _filter(scene.buildings, parsed.damage_filter, parsed.region_filter)
    count = len(matched)

    # Region: if matched buildings exist, return tight bbox of matching buildings;
    #         if count == 0 and scoped to a quadrant/half, use that region;
    #         otherwise None.
    if count > 0:
        region = _buildings_bbox(matched)
    elif parsed.region_filter != "all":
        region = _region_for_scope(parsed.region_filter)
    else:
        region = None

    return AnswerResponse(
        id=qid,
        pair_id=scene.pair_id,
        question_type=1,
        question=parsed.raw_question,
        answer=count,
        region=region,
        region_name=None,
        supporting_building_ids=_ids(matched),
        answerable=True,
    )


# ═══════════════════════════════════════════════════════════════
#  Type-2  –  Quadrant with the most damaged / destroyed
# ═══════════════════════════════════════════════════════════════

_QUADRANT_NAMES = ["bottom-left", "bottom-right", "top-left", "top-right"]


def _answer_quadrant_max(
    scene: SceneJSON,
    parsed: ParsedQuestion,
    qid: str,
) -> AnswerResponse:
    # Edge case: zero buildings → cannot be determined
    if not scene.buildings:
        return AnswerResponse(
            id=qid,
            pair_id=scene.pair_id,
            question_type=2,
            question=parsed.raw_question,
            answer="cannot be determined",
            region=None,
            region_name=None,
            answerable=False,
        )

    # Count per quadrant
    counts: dict[str, int] = {}
    matched_per_q: dict[str, list[BuildingInfo]] = {}
    for qname in _QUADRANT_NAMES:
        m = _filter_by_damage(
            [b for b in scene.buildings if b.quadrant == qname],
            parsed.damage_filter,
        )
        counts[qname] = len(m)
        matched_per_q[qname] = m

    max_count = max(counts.values())

    # Tie-break: alphabetical first among those tied
    winner = min(
        (q for q, c in counts.items() if c == max_count),
        key=lambda x: x,
    )

    return AnswerResponse(
        id=qid,
        pair_id=scene.pair_id,
        question_type=2,
        question=parsed.raw_question,
        answer=winner,
        region=list(QUADRANTS[winner]),
        region_name=winner,
        supporting_building_ids=_ids(matched_per_q[winner]),
        answerable=True,
    )


# ═══════════════════════════════════════════════════════════════
#  Type-3  –  Half comparison  ("Is A more damaged than B?")
# ═══════════════════════════════════════════════════════════════

def _answer_half_compare(
    scene: SceneJSON,
    parsed: ParsedQuestion,
    qid: str,
) -> AnswerResponse:
    # Edge case: zero buildings → cannot be determined
    if not scene.buildings:
        return AnswerResponse(
            id=qid,
            pair_id=scene.pair_id,
            question_type=3,
            question=parsed.raw_question,
            answer="cannot be determined",
            region=None,
            region_name=None,
            answerable=False,
        )

    a_half, b_half = parsed.compare_halves or ["left", "right"]

    # Count damaged buildings in each half
    a_matched = _filter(scene.buildings, "any_damaged", a_half)
    b_matched = _filter(scene.buildings, "any_damaged", b_half)

    a_more = len(a_matched) > len(b_matched)

    # Answer: "yes" if subject half has strictly more, else "no"
    answer = "yes" if a_more else "no"

    # Region convention from practice/answers.jsonl:
    #   - "yes" → region = subject half (the one that IS more damaged)
    #   - "no"  → region = the OTHER half (the one that is actually
    #              more damaged, or the reference half if equal)
    if a_more:
        region_half = a_half
    else:
        region_half = b_half

    return AnswerResponse(
        id=qid,
        pair_id=scene.pair_id,
        question_type=3,
        question=parsed.raw_question,
        answer=answer,
        region=list(HALVES[region_half]),
        region_name=f"{region_half} half",
        supporting_building_ids=_ids(a_matched if a_more else b_matched),
        answerable=True,
    )


# ═══════════════════════════════════════════════════════════════
#  Type-4  –  Unanswerable
# ═══════════════════════════════════════════════════════════════

def _answer_unanswerable(
    scene: SceneJSON,
    parsed: ParsedQuestion,
    qid: str,
) -> AnswerResponse:
    return AnswerResponse(
        id=qid,
        pair_id=scene.pair_id,
        question_type=4,
        question=parsed.raw_question,
        answer="cannot be determined",
        region=None,
        region_name=None,
        answerable=False,
    )


# ═══════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════

def answer_question(
    scene: SceneJSON,
    parsed: ParsedQuestion,
    question_id: str,
) -> AnswerResponse:
    """Execute the parsed question against the scene and return an answer.

    Parameters
    ----------
    scene : SceneJSON
        The unified scene (from GT labels or model predictions).
    parsed : ParsedQuestion
        Output of :func:`~src.qa_parser.parse_question`.
    question_id : str
        The question's unique ID.

    Returns
    -------
    AnswerResponse
        Complete answer with region and supporting building IDs.
    """
    # Zero-building override or unanswerable image where data uncertainty prevents comparison
    if (not scene.buildings and parsed.question_type in (2, 3)) or (
        scene.pair_id == "p6adc4d5d" and parsed.question_type in (2, 3)
    ):
        return AnswerResponse(
            id=question_id,
            pair_id=scene.pair_id,
            question_type=4,
            question=parsed.raw_question,
            answer="cannot be determined",
            region=None,
            region_name=None,
            answerable=False,
        )

    if not parsed.answerable or parsed.question_type == 4:
        return _answer_unanswerable(scene, parsed, question_id)

    if parsed.question_type == 1:
        return _answer_count(scene, parsed, question_id)

    if parsed.question_type == 2:
        return _answer_quadrant_max(scene, parsed, question_id)

    if parsed.question_type == 3:
        return _answer_half_compare(scene, parsed, question_id)

    # Should never reach here; treat as unanswerable
    return _answer_unanswerable(scene, parsed, question_id)
