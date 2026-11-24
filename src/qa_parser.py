"""
Rules-first question parser for PS04 Aerial View Image Reasoning.

Parses every question into a structured ``ParsedQuestion`` by matching
against the patterns in ``question_templates.json`` and known natural
variations from the practice set.  An optional LLM fallback stub is
included but disabled by default.

Parsing order:
  1. Try Type-2 (quadrant max) — starts with "Which quadrant"
  2. Try Type-3 (half comparison) — "Is the … half …" / "Does the … half …"
  3. Try Type-1 (count) — "How many …" / "Count the …"
  4. Fallback → Type-4 (unanswerable)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  ParsedQuestion — the structured representation of any question
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParsedQuestion:
    """Structured representation produced by the parser."""

    question_type: int
    """1 = count, 2 = quadrant max, 3 = half comparison, 4 = unanswerable."""

    damage_filter: str
    """
    One of:
      ``"no-damage"`` | ``"minor-damage"`` | ``"major-damage"`` |
      ``"destroyed"`` | ``"any_damaged"`` (minor+major+destroyed) |
      ``"all"`` (every building regardless of damage).
    """

    region_filter: str
    """
    ``"all"`` (whole image) or a quadrant / half name:
      ``"top-left"`` | ``"top-right"`` | ``"bottom-left"`` | ``"bottom-right"``
      | ``"top"`` | ``"bottom"`` | ``"left"`` | ``"right"``.
    """

    compare_axis: Optional[str] = None
    """``"quadrants"`` for type 2, ``"left_vs_right"`` / ``"top_vs_bottom"``
    for type 3, ``None`` otherwise."""

    compare_halves: Optional[list[str]] = None
    """For type 3: ``[subject_half, other_half]``, e.g. ``["left", "right"]``."""

    answerable: bool = True
    """``False`` only for type-4 questions (unanswerable from imagery)."""

    unanswerable_reason: Optional[str] = None
    """Human-readable reason when ``answerable`` is ``False``."""

    raw_question: str = ""
    """The original question text, kept for debugging / logging."""


# ═══════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════

# Level descriptions from question_templates.json → damage_filter key
_LEVEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Order matters: longer / more specific patterns first
    (re.compile(r"damaged buildings \(minor,? major or destroyed\)", re.I), "any_damaged"),
    (re.compile(r"buildings with major damage", re.I),                      "major-damage"),
    (re.compile(r"buildings with minor damage", re.I),                      "minor-damage"),
    (re.compile(r"undamaged buildings", re.I),                              "no-damage"),
    (re.compile(r"buildings with no[- ]damage", re.I),                      "no-damage"),
    (re.compile(r"destroyed buildings", re.I),                              "destroyed"),
    (re.compile(r"damaged buildings", re.I),                                "any_damaged"),
    (re.compile(r"buildings", re.I),                                        "all"),
]

# Scope / region patterns
_SCOPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"in the (top-left|top left)\s*quadrant",   re.I), "top-left"),
    (re.compile(r"in the (top-right|top right)\s*quadrant", re.I), "top-right"),
    (re.compile(r"in the (bottom-left|bottom left)\s*quadrant",  re.I), "bottom-left"),
    (re.compile(r"in the (bottom-right|bottom right)\s*quadrant", re.I), "bottom-right"),
    (re.compile(r"in the top half",    re.I), "top"),
    (re.compile(r"in the bottom half", re.I), "bottom"),
    (re.compile(r"in the left half",   re.I), "left"),
    (re.compile(r"in the right half",  re.I), "right"),
    (re.compile(r"in the image",       re.I), "all"),
]

# Half extraction for Type 3
_HALF_RE = re.compile(
    r"(?:is|does)\s+the\s+(top|bottom|left|right)\s+half\b"
    r".+?"
    r"(?:than|versus|vs\.?)\s+the\s+(top|bottom|left|right)\s+half",
    re.IGNORECASE,
)

# Type-4 keyword groups and their reasons
_TYPE4_INDICATORS: list[tuple[list[str], str]] = [
    (
        ["hospital", "school", "church", "mosque", "temple", "clinic",
         "warehouse", "factory", "office", "apartment", "store", "shop",
         "is a ", "type of building", "function of"],
        "Building function/type cannot be determined from satellite imagery.",
    ),
    (
        ["people", "person", "resident", "population", "lived in",
         "lives in", "living in", "casualties", "deaths", "injured",
         "evacuate", "survivors", "victims"],
        "Population/occupancy data cannot be determined from satellite imagery.",
    ),
    (
        ["rebuilt", "rebuild", "repair", "next year", "will need",
         "will be", "in the future", "planned", "going to"],
        "Future events/plans cannot be determined from satellite imagery.",
    ),
    (
        ["cost", "price", "value", "dollar", "money", "insurance",
         "economic", "financial"],
        "Financial information cannot be determined from satellite imagery.",
    ),
    (
        ["date", "when was", "what time", "what day", "what month",
         "what year", "taken on"],
        "Temporal metadata cannot be determined from satellite imagery alone.",
    ),
    (
        ["who ", "whose", "owner", "name of", "named",
         "government", "agency", "responsible"],
        "Identity/ownership information cannot be determined from satellite imagery.",
    ),
    (
        ["road", "infrastructure", "bridge", "power line", "water supply"],
        "Non-building infrastructure analysis is out of scope.",
    ),
]


def _detect_damage_filter(text: str) -> str:
    """Extract the damage-level filter from question text."""
    for pat, dmg in _LEVEL_PATTERNS:
        if pat.search(text):
            return dmg
    return "all"


def _detect_region_filter(text: str) -> str:
    """Extract the region/scope filter from question text."""
    for pat, region in _SCOPE_PATTERNS:
        if pat.search(text):
            return region
    return "all"


def _detect_type4_reason(text: str) -> Optional[str]:
    """Check if question contains type-4 (unanswerable) indicators."""
    text_lower = text.lower()
    for keywords, reason in _TYPE4_INDICATORS:
        for kw in keywords:
            if kw.lower() in text_lower:
                return reason
    return None


# ═══════════════════════════════════════════════════════════════
#  Type-specific parsers
# ═══════════════════════════════════════════════════════════════

def _parse_type1(q: str) -> ParsedQuestion:
    """Parse a counting question (Type 1)."""
    return ParsedQuestion(
        question_type=1,
        damage_filter=_detect_damage_filter(q),
        region_filter=_detect_region_filter(q),
        answerable=True,
        raw_question=q,
    )


def _parse_type2(q: str) -> ParsedQuestion:
    """Parse a quadrant-max question (Type 2)."""
    q_lower = q.lower()
    # "most destroyed buildings" (destroyed only) vs "most damaged buildings" (any)
    if "destroyed" in q_lower and "damaged" not in q_lower:
        dmg = "destroyed"
    else:
        dmg = "any_damaged"
    return ParsedQuestion(
        question_type=2,
        damage_filter=dmg,
        region_filter="all",
        compare_axis="quadrants",
        answerable=True,
        raw_question=q,
    )


def _parse_type3(q: str) -> ParsedQuestion:
    """Parse a half-comparison question (Type 3)."""
    m = _HALF_RE.search(q)
    if m:
        a_half = m.group(1).lower()   # subject: "Is the LEFT half more damaged…"
        b_half = m.group(2).lower()   # reference: "…than the RIGHT half?"
    else:
        # Fallback: extract any two half names in order
        halves_found = re.findall(r"\b(top|bottom|left|right)\s+half\b", q, re.I)
        if len(halves_found) >= 2:
            a_half = halves_found[0].lower()
            b_half = halves_found[1].lower()
        else:
            a_half, b_half = "left", "right"  # safe default

    if {a_half, b_half} <= {"left", "right"}:
        axis = "left_vs_right"
    elif {a_half, b_half} <= {"top", "bottom"}:
        axis = "top_vs_bottom"
    else:
        axis = f"{a_half}_vs_{b_half}"

    return ParsedQuestion(
        question_type=3,
        damage_filter="any_damaged",
        region_filter="all",
        compare_axis=axis,
        compare_halves=[a_half, b_half],
        answerable=True,
        raw_question=q,
    )


def _parse_type4(q: str) -> ParsedQuestion:
    """Parse / flag an unanswerable question (Type 4)."""
    reason = _detect_type4_reason(q)
    if reason is None:
        reason = "Question cannot be answered from satellite imagery alone."
    return ParsedQuestion(
        question_type=4,
        damage_filter="all",
        region_filter="all",
        answerable=False,
        unanswerable_reason=reason,
        raw_question=q,
    )


# ═══════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════

def parse_question(question: str) -> ParsedQuestion:
    """Parse a natural-language question into a structured :class:`ParsedQuestion`.

    The parser tries template patterns in order of specificity and falls
    back to Type-4 (unanswerable) if nothing matches.  It never calls an
    LLM — that path is behind :func:`parse_question_with_llm_fallback`.
    """
    q = question.strip()
    q_lower = q.lower()

    # ── Type 2: quadrant comparison ─────────────────────────────
    if q_lower.startswith("which quadrant"):
        return _parse_type2(q)

    # ── Type 3: half comparison ─────────────────────────────────
    # Must contain "half" and a comparison verb
    if "half" in q_lower and (
        q_lower.startswith("is the")
        or q_lower.startswith("does the")
    ):
        return _parse_type3(q)

    # ── Type 4: out-of-scope detection (before Type 1) ─────────
    # Some Type-4 Qs look like counting ("How many people lived…")
    type4_reason = _detect_type4_reason(q)
    if type4_reason is not None:
        # Double-check it's not a valid counting Q about buildings
        # e.g., "How many destroyed buildings are there?" is Type 1,
        # even though "destroyed" is in the type-4 keyword list.
        # → If the question matches a counting pattern AND refers
        #   to a known building damage level, treat it as Type 1.
        is_counting = q_lower.startswith("how many") or q_lower.startswith("count the")
        has_building_level = _detect_damage_filter(q) != "all" or "buildings" in q_lower
        # Type-4 keywords that NEVER co-occur with valid building Qs
        hard_type4 = any(
            kw in q_lower
            for kw in [
                "people", "person", "resident", "population",
                "lived", "lives", "living",
                "hospital", "school", "church", "mosque",
                "rebuilt", "rebuild", "repair", "next year",
                "will need", "will be", "cost", "price",
                "date", "when was", "who ", "owner",
                "casualties", "deaths", "injured",
            ]
        )
        if hard_type4:
            return _parse_type4(q)
        # Otherwise fall through to Type-1 parsing

    # ── Type 1: counting ────────────────────────────────────────
    if q_lower.startswith("how many") or q_lower.startswith("count the"):
        return _parse_type1(q)

    # ── Catch-all: Type 4 ──────────────────────────────────────
    return _parse_type4(q)


# ═══════════════════════════════════════════════════════════════
#  Optional LLM fallback (disabled by default)
# ═══════════════════════════════════════════════════════════════

_LLM_CACHE: dict[str, ParsedQuestion] = {}


def parse_question_with_llm_fallback(
    question: str,
    *,
    enable_llm: bool = False,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash",
) -> ParsedQuestion:
    """Try the rules-first parser; on ambiguity, optionally call an LLM.

    Parameters
    ----------
    question : str
        The raw question text.
    enable_llm : bool
        If ``True`` and the rules parser produces Type-4 for a question
        that *might* be answerable, call the structured-output LLM.
        **Default ``False``** — fully offline.
    api_key : str | None
        API key for the LLM provider (Gemini / Groq).
    model : str
        Model identifier for the LLM call.

    Returns
    -------
    ParsedQuestion
    """
    # Always try rules first
    parsed = parse_question(question)

    if parsed.question_type != 4 or not enable_llm:
        return parsed

    # Check cache
    cache_key = question.strip().lower()
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    # ── LLM structured-output call (stub) ──────────────────────
    # To activate, set enable_llm=True and provide api_key.
    # The LLM is asked to return JSON matching ParsedQuestion fields.
    # For now, this is a placeholder that returns the Type-4 parse.
    #
    #   from google import genai
    #   client = genai.Client(api_key=api_key)
    #   response = client.models.generate_content(
    #       model=model,
    #       contents=_build_llm_prompt(question),
    #       config={"response_mime_type": "application/json", ...},
    #   )
    #   parsed = _parse_llm_response(response.text, question)
    #   _LLM_CACHE[cache_key] = parsed

    return parsed
