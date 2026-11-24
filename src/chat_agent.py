"""
chat_agent.py: Grounded Natural-Language <-> Official JSON Translator for PS04 Disaster VQA.

Features:
  - Step A (NL -> Intent JSON):
    Uses Groq's OpenAI-compatible endpoint (model `llama-3.1-8b-instant`, response_format={"type": "json_object"})
    or Gemini, with a robust offline few-shot regex fallback that correctly handles:
      * "total number of buildings" -> question_type: 1, damage_filter: "all", region: "full"
      * "highlight the quadrant with most damaged buildings" -> question_type: 2, damage_filter: "damaged"
  - Step B (Deterministic CV Execution):
    Executes intent strictly over SceneJSON via spatial_engine.py so all counts and regions come from our models.
  - Step C (JSON -> Grounded Conversational Reply):
    Generates natural, helpful emergency responder responses via Groq (llama-3.1-8b-instant, temp=0.3)
    conditioned strictly on verified JSON, with grounded deterministic fallback.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from pydantic import BaseModel, Field, field_validator

from .qa_parser import ParsedQuestion, parse_question
from .schema import AnswerResponse, BuildingInfo, DamageLevel, SceneJSON
from .spatial_engine import answer_question


# ═══════════════════════════════════════════════════════════════
#  Automatic Environment Auto-Loader (.env & api.txt)
# ═══════════════════════════════════════════════════════════════

def _load_env_file() -> None:
    """Load key-value pairs from .env or api.txt in project root into os.environ."""
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Check .env in repository root
    env_file = repo_root / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and not os.environ.get(k):
                            os.environ[k] = v
        except Exception:
            pass

    # 2. Check api.txt fallback if GROQ_API_KEY is not already set
    api_txt = repo_root / "api.txt"
    if api_txt.exists() and not os.environ.get("GROQ_API_KEY"):
        try:
            with open(api_txt, "r", encoding="utf-8") as f:
                raw_key = f.read().strip()
                if raw_key and raw_key.startswith("gsk_"):
                    os.environ["GROQ_API_KEY"] = raw_key
        except Exception:
            pass


_load_env_file()


# ═══════════════════════════════════════════════════════════════
#  Structured Intent Schema
# ═══════════════════════════════════════════════════════════════

class ChatIntent(BaseModel):
    """Structured query intent extracted from natural language input."""
    question_type: int = Field(..., description="0=Greeting, 1=Count, 2=Quadrant Max, 3=Half Comparison, 4=Unanswerable")
    damage_filter: str = Field(
        default="all",
        description="'all', 'damaged', 'no-damage', 'minor-damage', 'major-damage', or 'destroyed'",
    )
    region: str = Field(
        default="full",
        description="'full', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'top-half', 'bottom-half', 'left-half', 'right-half'",
    )
    normalized_question: Optional[str] = Field(default=None, description="Cleaned canonical question text")
    is_out_of_scope: bool = Field(default=False, description="True if query asks for non-visual attributes (hospital, casualties, population, costs, date, rebuilding)")
    is_greeting: bool = Field(default=False, description="True for greetings, pleasantries, help, or intro queries")
    explanation: Optional[str] = Field(default=None, description="Reasoning note for unanswerable queries")

    @field_validator("damage_filter", mode="before")
    @classmethod
    def normalize_damage_filter(cls, v: Any) -> str:
        """Coerce list or raw string to canonical damage filter string."""
        if isinstance(v, list):
            if not v:
                return "all"
            if len(v) == 1:
                v = v[0]
            elif "no-damage" not in v:
                return "damaged"
            else:
                return "all"
        v_str = str(v).lower().strip().replace("_", "-")
        if v_str in ("damaged", "all-damaged", "any-damaged", "damage", "impacted", "affected"):
            return "damaged"
        if v_str in ("total", "any", "every", "all-buildings"):
            return "all"
        if v_str in ("no-damage", "undamaged", "intact"):
            return "no-damage"
        if v_str in ("minor-damage", "minor"):
            return "minor-damage"
        if v_str in ("major-damage", "major"):
            return "major-damage"
        if v_str in ("destroyed", "flattened", "collapsed"):
            return "destroyed"
        return "all"

    @field_validator("region", mode="before")
    @classmethod
    def normalize_region(cls, v: Any) -> str:
        """Coerce region text to valid region string."""
        v_str = str(v).lower().strip().replace(" ", "-")
        if v_str in ("all", "entire", "image", "scene", "whole"):
            return "full"
        if v_str in ("top-left", "top-right", "bottom-left", "bottom-right", "top-half", "bottom-half", "left-half", "right-half", "full"):
            return v_str
        if v_str == "left":
            return "left-half"
        if v_str == "right":
            return "right-half"
        if v_str == "top":
            return "top-half"
        if v_str == "bottom":
            return "bottom-half"
        return "full"


# ═══════════════════════════════════════════════════════════════
#  Step A: Natural Language -> Structured Intent JSON
# ═══════════════════════════════════════════════════════════════

# Out-of-scope trigger patterns: Strictly non-visual attributes
OUT_OF_SCOPE_PATTERNS = [
    r"\b(hospital|clinic|doctor|medical|health)\b",
    r"\b(school|university|college|classroom|student|teacher)\b",
    r"\b(people|residents?|citizens?|casualties|deaths?|fatalit(y|ies)|injured|injuries|lives?|survivors?|evacuat(ed|ion)?)\b",
    r"\b(rebuilt?|rebuilding|reconstruction|future|next year|repair costs?|monetary|dollar|\$|budget|insur(ance)?)\b",
    r"\b(date|time of day|when was|taken on|calendar|month|year|timestamp|clock)\b",
    r"\b(who lives|who owns|owner|landlord|address|street name|resident name|family|occupant)\b",
    r"\b(fire station|police|military|government office|post office)\b",
    r"\b(earthquake magnitude|richter|wind speed|category of hurricane)\b",
]

GROQ_INTENT_SYSTEM_PROMPT = """You are a satellite disaster visual question-answering query intent parser.
Convert the user's natural language question into strict JSON with the following schema:
{
  "question_type": 0 | 1 | 2 | 3 | 4,
  "damage_filter": "all" | "damaged" | "no-damage" | "minor-damage" | "major-damage" | "destroyed",
  "region": "full" | "top-left" | "top-right" | "bottom-left" | "bottom-right" | "top-half" | "bottom-half" | "left-half" | "right-half",
  "is_out_of_scope": false | true,
  "is_greeting": false | true
}

Rules:
0. is_greeting & question_type=0:
   - True for greetings (hello, hi, hey, good morning), pleasantries (thanks, thank you, cool, awesome), or general queries about capabilities (who are you, what can you do, what is this app, help).
   - In this case, set is_greeting=true, question_type=0, is_out_of_scope=false.

1. question_type:
   - 0: Greeting, pleasantry, or capability inquiry.
   - 1: Counting buildings (e.g. "how many", "count", "total number of buildings", "number of houses", "how many buildings are there").
   - 2: Quadrant comparison (e.g. "which quadrant has the most damaged buildings", "highlight the quadrant with most damaged buildings", "which area was hit hardest", "where is damage concentrated").
   - 3: Half comparison (e.g. "is the left half more damaged than the right half", "does the top half have more damaged buildings than the bottom half").
   - 4: Unanswerable from overhead imagery (asking for hospital/school identity, casualties/deaths/injuries, population/residents, repair costs/money/budget, disaster date/time, future rebuilding plans).

2. damage_filter:
   - "all": Any query asking for TOTAL buildings, houses, or structures without filtering by damage (e.g. "total number of buildings", "how many buildings are there").
   - "damaged": Asking for damaged, destroyed, collapsed, or affected structures in general (minor, major, or destroyed).
   - "destroyed": Completely destroyed, flattened, leveled, demolished, or collapsed.
   - "major-damage": Major structural damage or heavy damage.
   - "minor-damage": Minor damage or slight damage.
   - "no-damage": Undamaged, intact, standing, or survived structures.

3. region:
   - "full": Whole scene / image.
   - "top-left", "top-right", "bottom-left", "bottom-right": Quadrants.
   - "top-half", "bottom-half", "left-half", "right-half": Halves.

4. is_out_of_scope:
   - True ONLY when asking about non-visual attributes (hospitals, schools, casualties, resident names, monetary costs, future rebuilding, exact dates).
   - False for greetings and for all physical building counts, damage assessments, quadrant rankings, and half comparisons.

Examples:
- "hello" -> {"question_type": 0, "damage_filter": "all", "region": "full", "is_out_of_scope": false, "is_greeting": true}
- "what can you do?" -> {"question_type": 0, "damage_filter": "all", "region": "full", "is_out_of_scope": false, "is_greeting": true}
- "total number of buildings" -> {"question_type": 1, "damage_filter": "all", "region": "full", "is_out_of_scope": false, "is_greeting": false}
- "highlight the quadrant with most damaged buildings" -> {"question_type": 2, "damage_filter": "damaged", "region": "full", "is_out_of_scope": false, "is_greeting": false}
- "how many houses were completely destroyed on the left half" -> {"question_type": 1, "damage_filter": "destroyed", "region": "left-half", "is_out_of_scope": false, "is_greeting": false}
- "is the top half more damaged than the bottom half" -> {"question_type": 3, "damage_filter": "damaged", "region": "top-half", "is_out_of_scope": false, "is_greeting": false}
- "which building in the image is a hospital" -> {"question_type": 4, "damage_filter": "all", "region": "full", "is_out_of_scope": true, "is_greeting": false}
- "how many casualties were reported from this disaster" -> {"question_type": 4, "damage_filter": "all", "region": "full", "is_out_of_scope": true, "is_greeting": false}
"""


def _offline_rules_parser(question: str) -> ChatIntent:
    """Robust, 100% offline regex & keyword intent parser."""
    q_clean = question.strip()
    q_lower = q_clean.lower()

    # 0. Check for Greeting / Capability inquiries
    greeting_patterns = [
        r"^(hello|hi|hey|howdy|greetings|good\s+(morning|afternoon|evening))\b",
        r"\b(who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+this|help|how\s+does\s+this\s+work)\b",
        r"^(thanks|thank\s+you|awesome|cool|great)\b",
    ]
    for pattern in greeting_patterns:
        if re.search(pattern, q_lower):
            return ChatIntent(
                question_type=0,
                damage_filter="all",
                region="full",
                normalized_question=q_clean,
                is_out_of_scope=False,
                is_greeting=True,
            )

    # 1. Check for Out-of-Scope (Type 4) triggers strictly
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, q_lower):
            return ChatIntent(
                question_type=4,
                damage_filter="all",
                region="full",
                normalized_question=q_clean,
                is_out_of_scope=True,
                explanation="Query asks for non-visual information (facility function, demographics, casualties, dates, or financial costs) that cannot be determined from aerial imagery alone.",
            )

    # 2. Extract Spatial Region
    region = "full"
    if re.search(r"\b(top[\s\-_]?left|upper[\s\-_]?left|north[\s\-_]?west)\b", q_lower):
        region = "top-left"
    elif re.search(r"\b(top[\s\-_]?right|upper[\s\-_]?right|north[\s\-_]?east)\b", q_lower):
        region = "top-right"
    elif re.search(r"\b(bottom[\s\-_]?left|lower[\s\-_]?left|south[\s\-_]?west)\b", q_lower):
        region = "bottom-left"
    elif re.search(r"\b(bottom[\s\-_]?right|lower[\s\-_]?right|south[\s\-_]?east)\b", q_lower):
        region = "bottom-right"
    elif re.search(r"\b(left[\s\-_]?(side|half)|on the left|western half)\b", q_lower):
        region = "left-half"
    elif re.search(r"\b(right[\s\-_]?(side|half)|on the right|eastern half)\b", q_lower):
        region = "right-half"
    elif re.search(r"\b(top[\s\-_]?(side|half)|upper half|on top|northern half)\b", q_lower):
        region = "top-half"
    elif re.search(r"\b(bottom[\s\-_]?(side|half)|lower half|on bottom|southern half)\b", q_lower):
        region = "bottom-half"

    # 3. Detect Question Type
    is_quadrant_query = (
        bool(re.search(r"\b(which|what|show|highlight|find|locate)\b.*\b(quadrant|area|sector|zone)\b", q_lower))
        or "quadrant with most" in q_lower
        or "most damaged quadrant" in q_lower
        or "hit hardest" in q_lower
        or "worst damage" in q_lower
        or "largest number of damaged" in q_lower
    )

    is_half_query = (
        bool(re.search(r"\b(more damaged than|greater damage|worse than)\b", q_lower))
        or (bool(re.search(r"\b(is the|does the)\b", q_lower)) and bool(re.search(r"\b(half|side)\b", q_lower)))
    )

    if is_quadrant_query:
        q_type = 2
    elif is_half_query:
        q_type = 3
    else:
        q_type = 1

    # 4. Extract Damage Filter
    is_total_query = bool(
        re.search(r"\b(total|all)\s+(number\s+of\s+)?(buildings|houses|structures)\b", q_lower)
        or re.search(r"\bhow many\s+(total\s+)?(buildings|houses|structures)\s+(are|in)\b", q_lower)
        or q_lower.strip() in ("total buildings", "all buildings", "count buildings", "number of buildings")
    )

    if is_total_query:
        damage_filter = "all"
    elif re.search(r"\b(completely destroyed|destroyed|flattened|collapsed|demolished|razed|leveled)\b", q_lower):
        damage_filter = "destroyed"
    elif re.search(r"\b(major[\s\-_]damage|severely damaged|heavy damage|serious damage)\b", q_lower):
        damage_filter = "major-damage"
    elif re.search(r"\b(minor[\s\-_]damage|partially damaged|slight damage|light damage)\b", q_lower):
        damage_filter = "minor-damage"
    elif re.search(r"\b(undamaged|intact|standing|unharmed|no[\s\-_]damage|survived)\b", q_lower):
        damage_filter = "no-damage"
    elif re.search(r"\b(damaged|affected|impacted|hit hardest|destruction)\b", q_lower):
        damage_filter = "damaged"
    else:
        # Default based on question type
        damage_filter = "damaged" if q_type in (2, 3) else "all"

    return ChatIntent(
        question_type=q_type,
        damage_filter=damage_filter,
        region=region,
        normalized_question=q_clean,
        is_out_of_scope=False,
    )


GROQ_CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama3-8b-8192",
]
_CACHED_GROQ_MODEL: Optional[str] = None


def get_active_groq_model(api_key: Optional[str] = None) -> str:
    """Discover or return the active, verified Groq chat model for this API key."""
    global _CACHED_GROQ_MODEL
    if _CACHED_GROQ_MODEL:
        return _CACHED_GROQ_MODEL

    key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
    if not key:
        return "llama-3.3-70b-versatile"

    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PS04DisasterAgent/1.0",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=4.0)
        if resp.status_code == 200:
            available = [m["id"] for m in resp.json().get("data", [])]
            for cand in GROQ_CANDIDATE_MODELS:
                if cand in available:
                    _CACHED_GROQ_MODEL = cand
                    return cand
            chat_models = [m for m in available if "whisper" not in m and "guard" not in m]
            if chat_models:
                _CACHED_GROQ_MODEL = chat_models[0]
                return chat_models[0]
    except Exception:
        pass

    _CACHED_GROQ_MODEL = "llama-3.3-70b-versatile"
    return _CACHED_GROQ_MODEL


def _llm_groq_intent_parser(question: str, api_key: str) -> Optional[ChatIntent]:
    """Call Groq API with JSON mode and multi-model failover."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PS04DisasterAgent/1.0",
        "Content-Type": "application/json",
    }
    primary = get_active_groq_model(api_key)
    candidates = [primary] + [m for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"] if m != primary]

    for model in candidates:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": GROQ_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 500,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=6.0)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed_json = json.loads(content)
                parsed_json["normalized_question"] = question
                return ChatIntent.model_validate(parsed_json)
        except Exception:
            pass
    return None


def _llm_greeting_reply_generator(
    question: str,
    scene: SceneJSON,
    api_key: str,
) -> Optional[str]:
    """Generate a warm, intelligent conversational greeting acknowledging current loaded scene."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PS04DisasterAgent/1.0",
        "Content-Type": "application/json",
    }
    bldgs = scene.buildings
    n_undamaged = sum(1 for b in bldgs if b.damage == "no-damage")
    n_minor = sum(1 for b in bldgs if b.damage == "minor-damage")
    n_major = sum(1 for b in bldgs if b.damage == "major-damage")
    n_destroyed = sum(1 for b in bldgs if b.damage == "destroyed")

    sys_greet = (
        "You are an intelligent aerial disaster response AI assistant.\n"
        "A responder or emergency manager has greeted you or asked about your capabilities.\n"
        "Respond warmly, conversationally, and concisely in 2-3 sentences.\n"
        "Acknowledge the current scene context provided.\n"
        "Do NOT output robotic metadata or pixel coordinates.\n"
        "Invite them to ask questions about building counts, damage tiers, quadrant hotspots, or cardinal halves."
    )
    user_greet = (
        f"User Message: \"{question}\"\n"
        f"Current Scene Context:\n"
        f"- Scene ID: {scene.pair_id}\n"
        f"- Disaster: {scene.disaster or scene.disaster_type or 'disaster incident'}\n"
        f"- Total Detected Buildings: {len(bldgs)}\n"
        f"- Breakdown: {n_undamaged} undamaged, {n_minor} minor, {n_major} major, {n_destroyed} destroyed"
    )

    primary = get_active_groq_model(api_key)
    candidates = [primary] + [m for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"] if m != primary]

    for model in candidates:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_greet},
                {"role": "user", "content": user_greet},
            ],
            "temperature": 0.7,
            "max_tokens": 500,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=6.0)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
    return None


def nl_to_intent(question: str, groq_key: Optional[str] = None) -> Tuple[ChatIntent, str]:
    """Convert free-form natural language query into strict ChatIntent JSON.

    Returns:
        (ChatIntent, backend_name)
    """
    # 1. Check passed key or environment key for Groq
    api_key = (groq_key or os.environ.get("GROQ_API_KEY", "")).strip()
    if api_key:
        intent = _llm_groq_intent_parser(question, api_key)
        if intent is not None:
            active_model = get_active_groq_model(api_key)
            return intent, f"Groq ({active_model})"

    # 2. Check Gemini API key
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            prompt = f"{GROQ_INTENT_SYSTEM_PROMPT}\nUser Question: {question}\nReturn JSON only."
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json"},
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=4.0)
            if resp.status_code == 200:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                parsed_json = json.loads(text)
                parsed_json["normalized_question"] = question
                return ChatIntent.model_validate(parsed_json), "Gemini-2.5-Flash"
        except Exception:
            pass

    # 3. Built-in Deterministic Parser (always available, 0ms, 100% offline)
    intent = _offline_rules_parser(question)
    return intent, "Deterministic Offline (0ms)"


# ═══════════════════════════════════════════════════════════════
#  Step B: Deterministic Execution over SceneJSON
# ═══════════════════════════════════════════════════════════════

def execute_intent_on_scene(
    scene: SceneJSON,
    intent: ChatIntent,
    question_id: str = "chat_q",
) -> AnswerResponse:
    """Execute structured intent strictly using our spatial geometry engine.

    Guarantees that all rooftop counts, quadrant rankings, and bounding regions
    derive directly from Stage 1/2 computer vision outputs.
    """
    if intent.is_out_of_scope or intent.question_type == 4:
        # Strict out-of-scope response
        return AnswerResponse(
            id=question_id,
            pair_id=scene.pair_id,
            question=intent.normalized_question or "User query",
            question_type=4,
            answer="cannot be determined",
            answerable=False,
            region=None,
            region_name=None,
            supporting_building_ids=[],
            confidence=1.0,
        )

    # Map region
    region_conv = {
        "full": "all",
        "top-left": "top-left",
        "top-right": "top-right",
        "bottom-left": "bottom-left",
        "bottom-right": "bottom-right",
        "top-half": "top",
        "bottom-half": "bottom",
        "left-half": "left",
        "right-half": "right",
    }
    rf = region_conv.get(intent.region, "all")

    # Map damage_filter
    df_raw = intent.damage_filter
    if df_raw in ("all", "all_buildings", "total"):
        df = "all"
    elif df_raw in ("damaged", "all_damaged", "any_damaged"):
        df = "any_damaged"
    elif df_raw == "no-damage":
        df = "no-damage"
    elif df_raw == "destroyed":
        df = "destroyed"
    elif df_raw == "minor-damage":
        df = "minor-damage"
    elif df_raw == "major-damage":
        df = "major-damage"
    else:
        df = "all" if intent.question_type == 1 else "any_damaged"

    # Setup comparison axes for Types 2 and 3
    cmp_axis = None
    cmp_halves = None
    if intent.question_type == 2:
        cmp_axis = "quadrants"
    elif intent.question_type == 3:
        q_low = (intent.normalized_question or "").lower()
        if "top" in q_low and "bottom" in q_low:
            cmp_axis = "top_vs_bottom"
            if q_low.find("top") < q_low.find("bottom"):
                cmp_halves = ["top", "bottom"]
            else:
                cmp_halves = ["bottom", "top"]
        else:
            cmp_axis = "left_vs_right"
            if q_low.find("left") < q_low.find("right"):
                cmp_halves = ["left", "right"]
            else:
                cmp_halves = ["right", "left"]

    parsed = ParsedQuestion(
        question_type=intent.question_type,
        damage_filter=df,
        region_filter=rf,
        compare_axis=cmp_axis,
        compare_halves=cmp_halves,
        answerable=True,
    )

    # Deterministic spatial arithmetic
    return answer_question(scene, parsed, question_id=question_id)


# ═══════════════════════════════════════════════════════════════
#  Scene Summary Generator (~60 tokens for analytical QA)
# ═══════════════════════════════════════════════════════════════

def compute_scene_summary(scene: SceneJSON) -> Dict[str, Any]:
    """Deterministically compute a compact summary dictionary (~60 tokens) from verified CV scene.

    Returns:
        - total_buildings: int
        - by_damage_tier: {"no-damage": int, "minor-damage": int, "major-damage": int, "destroyed": int}
        - total_damaged: int (minor + major + destroyed)
        - by_quadrant: {"top-left": {"undamaged": int, "damaged": int}, ...}
        - by_half: {"top-half": {"undamaged": int, "damaged": int}, ...}
    """
    bldgs = scene.buildings
    total_buildings = len(bldgs)

    # 1. Damage tier breakdown
    by_damage_tier = {
        "no-damage": 0,
        "minor-damage": 0,
        "major-damage": 0,
        "destroyed": 0,
    }
    for b in bldgs:
        d = b.damage
        if d in by_damage_tier:
            by_damage_tier[d] += 1
        elif d in ("undamaged", "no_damage"):
            by_damage_tier["no-damage"] += 1
        else:
            by_damage_tier["minor-damage"] += 1

    total_damaged = (
        by_damage_tier["minor-damage"]
        + by_damage_tier["major-damage"]
        + by_damage_tier["destroyed"]
    )

    # 2. By quadrant: counts of undamaged and damaged
    by_quadrant = {
        "top-left": {"undamaged": 0, "damaged": 0},
        "top-right": {"undamaged": 0, "damaged": 0},
        "bottom-left": {"undamaged": 0, "damaged": 0},
        "bottom-right": {"undamaged": 0, "damaged": 0},
    }
    for b in bldgs:
        q = b.quadrant if (b.quadrant and b.quadrant in by_quadrant) else "top-left"
        cat = "undamaged" if b.damage == "no-damage" else "damaged"
        by_quadrant[q][cat] += 1

    # 3. By half: counts of undamaged and damaged for top-half, bottom-half, left-half, right-half
    by_half = {
        "top-half": {"undamaged": 0, "damaged": 0},
        "bottom-half": {"undamaged": 0, "damaged": 0},
        "left-half": {"undamaged": 0, "damaged": 0},
        "right-half": {"undamaged": 0, "damaged": 0},
    }
    for b in bldgs:
        cat = "undamaged" if b.damage == "no-damage" else "damaged"
        # Vertical half
        h_vert = b.halves[0] if (b.halves and len(b.halves) > 0) else ("top" if b.center[1] < 512 else "bottom")
        if h_vert == "top":
            by_half["top-half"][cat] += 1
        else:
            by_half["bottom-half"][cat] += 1

        # Horizontal half
        h_horiz = b.halves[1] if (b.halves and len(b.halves) > 1) else ("left" if b.center[0] < 512 else "right")
        if h_horiz == "left":
            by_half["left-half"][cat] += 1
        else:
            by_half["right-half"][cat] += 1

    return {
        "total_buildings": total_buildings,
        "by_damage_tier": by_damage_tier,
        "total_damaged": total_damaged,
        "by_quadrant": by_quadrant,
        "by_half": by_half,
    }


# ═══════════════════════════════════════════════════════════════
#  Step C: Official JSON + Scene Summary -> Grounded NL Reply
# ═══════════════════════════════════════════════════════════════

def _llm_natural_reply_generator(
    question: str,
    ans_resp: AnswerResponse,
    intent: ChatIntent,
    scene: SceneJSON,
    api_key: str,
    scene_summary: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Generate a conversational, natural responder reply using Groq with multi-model failover."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PS04DisasterAgent/1.0",
        "Content-Type": "application/json",
    }

    if scene_summary is None:
        scene_summary = compute_scene_summary(scene)

    official_competition_json = {
        "question_id": ans_resp.id,
        "pair_id": ans_resp.pair_id,
        "question_type": ans_resp.question_type,
        "answer": ans_resp.answer,
        "answerable": ans_resp.answerable,
        "region": ans_resp.region,
    }

    system_prompt = (
        "You are an expert aerial disaster intelligence specialist communicating directly with emergency commanders.\n"
        "You have access to two verified data objects produced by our computer-vision models:\n"
        "1. `official_competition_json`: The verified competition answer and region for the primary question.\n"
        "2. `scene_summary`: The complete verified building breakdown across damage tiers, quadrants, and halves.\n\n"
        "STRICT GROUNDING & REASONING RULES:\n"
        "- Use ONLY the exact numbers provided in `official_competition_json` and `scene_summary`. NEVER invent, extrapolate, or estimate figures.\n"
        "- If the user asks for a ratio, percentage, proportion, or multi-part comparison (e.g. 'calculate the ratio of undamaged buildings to damaged buildings'), "
        "calculate it accurately from `scene_summary` and show the numbers clearly (e.g. 'There are 142 undamaged buildings and 83 damaged buildings, giving a ratio of 1.71 to 1').\n"
        "- If answering a quadrant or directional comparison, state the winning region clearly.\n"
        "- Only say 'cannot be determined' if the user asks about non-visual facts (like hospitals, casualties, cost, or disaster cause) where `answerable` is false.\n"
        "- Speak naturally, concisely, and professionally like a human remote-sensing analyst. Never recite raw numeric pixel coordinate arrays like [818, 308, 855, 346] - describe locations in plain English (e.g. 'upper-right quadrant').\n"
        "- Output direct natural text only."
    )

    user_content = (
        f"User Question: \"{question}\"\n\n"
        f"official_competition_json:\n{json.dumps(official_competition_json, indent=2)}\n\n"
        f"scene_summary:\n{json.dumps(scene_summary, indent=2)}"
    )

    primary = get_active_groq_model(api_key)
    candidates = [primary] + [m for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"] if m != primary]

    for model in candidates:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=6.0)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
    return None


def format_grounded_reply(
    ans_resp: AnswerResponse,
    intent: ChatIntent,
    scene: SceneJSON,
    question: str = "",
    groq_key: Optional[str] = None,
    scene_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Produce a concise, human-friendly response grounded strictly in the official AnswerResponse and scene_summary."""
    if scene_summary is None:
        scene_summary = compute_scene_summary(scene)

    # 1. If Groq API key is present, try LLM conversational response
    api_key = (groq_key or os.environ.get("GROQ_API_KEY", "")).strip()
    if api_key and question:
        llm_reply = _llm_natural_reply_generator(
            question=question,
            ans_resp=ans_resp,
            intent=intent,
            scene=scene,
            api_key=api_key,
            scene_summary=scene_summary,
        )
        if llm_reply:
            return llm_reply

    # 2. Deterministic Grounded Formatter (Zero-Hallucination Fallback)
    ans = ans_resp.answer
    q_type = ans_resp.question_type

    # Analytical ratio / percentage fallback if LLM is offline
    q_lower = question.lower()
    if any(k in q_lower for k in ("ratio", "percentage", "percent", "proportion")) and scene_summary:
        u = scene_summary["by_damage_tier"].get("no-damage", 0)
        d = scene_summary["total_damaged"]
        if "undamaged" in q_lower and "damaged" in q_lower:
            r_val = round(u / max(d, 1), 2) if d > 0 else "N/A"
            return (
                f"📊 **Ratio Analysis:** Verified **{u}** undamaged to **{d}** damaged structures "
                f"(**{r_val}:1** ratio across {scene_summary['total_buildings']} total buildings)."
            )

    # Format human damage description
    if intent.damage_filter == "all":
        df_desc = "total buildings (all damage tiers combined)"
    elif intent.damage_filter == "destroyed":
        df_desc = "completely destroyed buildings"
    elif intent.damage_filter == "no-damage":
        df_desc = "undamaged structures"
    elif intent.damage_filter == "minor-damage":
        df_desc = "buildings with minor damage"
    elif intent.damage_filter == "major-damage":
        df_desc = "buildings with major structural damage"
    else:
        df_desc = "damaged buildings (minor, major or destroyed)"

    # Format region description
    reg_desc = "across the full scene"
    if intent.region != "full":
        reg_desc = f"in the {intent.region.replace('-', ' ')}"

    # Type 4: Unanswerable
    if q_type == 4 or not ans_resp.answerable:
        return (
            "🚫 **Cannot be determined from satellite imagery.**\n\n"
            "Overhead satellite imaging detects physical rooftop geometry and damage tiers, but **cannot identify building functions** "
            "(such as hospitals or schools), nor assess **casualties, resident populations, repair costs, or historical dates**.\n\n"
            "*Competition Verdict:* `cannot be determined` (`answerable: false`)."
        )

    # Type 1: Counting
    if q_type == 1:
        count = ans
        supp_n = len(ans_resp.supporting_building_ids)
        total_in_scene = len(scene.buildings)
        if count == 0:
            return (
                f"🔍 **0 Detected**: No {df_desc} were detected {reg_desc}.\n\n"
                f"*Verification:* Total structures analyzed across scene: **{total_in_scene}**."
            )
        return (
            f"📊 **Count: {count}**\n\n"
            f"Aerial instance segmentation detected **{count}** {df_desc} {reg_desc}. "
            f"The spatial engine verified **{supp_n}** rooftop footprints matching this query (out of **{total_in_scene}** total structures)."
        )

    # Type 2: Quadrant Comparison
    if q_type == 2:
        quad = str(ans)
        supp_n = len(ans_resp.supporting_building_ids)
        return (
            f"🧭 **Most Damaged Quadrant: {quad.upper()}**\n\n"
            f"Spatial density aggregation identified the **{quad}** quadrant as containing the highest concentration of {df_desc} "
            f"with **{supp_n}** affected structures."
        )

    # Type 3: Half Comparison
    if q_type == 3:
        verdict = str(ans).upper()
        reg_name = ans_resp.region_name or "highlighted half"
        return (
            f"⚖️ **Comparison Verdict: {verdict}**\n\n"
            f"Comparative analysis across cardinal planes concluded **{verdict}**. "
            f"The dominant affected region is the **{reg_name}**."
        )

    return f"**Answer:** `{ans}` (Region: `{ans_resp.region}`)"


# ═══════════════════════════════════════════════════════════════
#  Public Top-Level Chat Agent API
# ═══════════════════════════════════════════════════════════════

def ask_chat_agent(
    question: str,
    scene: SceneJSON,
    question_id: str = "chat_msg",
    groq_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full 3-step conversational pipeline:

      1. Natural Language -> Structured Intent JSON (via Groq or deterministic rules)
      2. Intent -> Deterministic Computer Vision Execution over SceneJSON
      3. Official AnswerResponse + Scene Summary -> Grounded Human-Friendly Response
    """
    # Deterministic scene summary (~60 tokens)
    scene_summary = compute_scene_summary(scene)

    # Step A: Intent Extraction
    intent, backend = nl_to_intent(question, groq_key=groq_key)

    # Conversational Greeting / Capability Handling
    if intent.is_greeting:
        api_key = (groq_key or os.environ.get("GROQ_API_KEY", "")).strip()
        greeting_text = None
        if api_key:
            greeting_text = _llm_greeting_reply_generator(question, scene, api_key)
        if not greeting_text:
            bldgs = scene.buildings
            greeting_text = (
                f"👋 **Hello!** I am your Aerial Disaster Response Assistant.\n\n"
                f"Currently analyzing Scene `{scene.pair_id}` ({scene.disaster or 'disaster incident'}). "
                f"Our computer vision models localized **{len(bldgs)}** total building structures.\n\n"
                f"You can ask me questions about **building counts** by damage tier (undamaged, minor, major, destroyed), "
                f"**damage ratios/percentages**, **quadrant triage**, or **cardinal comparisons** (comparing halves of the image)."
            )
        ans_resp = AnswerResponse(
            id=question_id,
            pair_id=scene.pair_id,
            question=question,
            question_type=1,
            answer=len(scene.buildings),
            answerable=True,
            region=[0, 0, 1024, 1024],
            region_name="Full Scene",
            supporting_building_ids=[b.id for b in scene.buildings],
        )
        return {
            "natural_response": greeting_text,
            "intent": intent.model_dump(),
            "backend": backend,
            "answer_response": ans_resp,
            "submission_json": {
                "id": question_id,
                "answer": f"{len(scene.buildings)} total buildings in scene",
                "region": [0, 0, 1024, 1024],
            },
            "full_json": {"id": question_id, "type": "greeting", "answer": f"{len(scene.buildings)} buildings"},
            "supporting_building_ids": ans_resp.supporting_building_ids,
            "region": [0, 0, 1024, 1024],
            "scene_summary": scene_summary,
        }

    # Step B: Deterministic Spatial QA Execution
    ans_resp = execute_intent_on_scene(scene, intent, question_id=question_id)

    # Step C: Grounded Response Formatting (with both official JSON and scene summary)
    natural_reply = format_grounded_reply(
        ans_resp=ans_resp,
        intent=intent,
        scene=scene,
        question=question,
        groq_key=groq_key,
        scene_summary=scene_summary,
    )

    return {
        "natural_response": natural_reply,
        "intent": intent.model_dump(),
        "backend": backend,
        "answer_response": ans_resp,
        "submission_json": ans_resp.to_submission(),
        "full_json": ans_resp.to_full_dict(),
        "supporting_building_ids": ans_resp.supporting_building_ids,
        "region": ans_resp.region,
        "scene_summary": scene_summary,
    }
