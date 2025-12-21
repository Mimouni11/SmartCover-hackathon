"""
Router Node - Fast keyword-based claim detection and routing.
First node in the pipeline.
"""

import unicodedata
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, AIMessage

from .state import ClaimState


def _norm(s: str) -> str:
    """Normalize text: lowercase + remove accents."""
    s = s.lower()
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    return s


def _latest_user_text(state: ClaimState) -> str:
    """Get the last user message from state."""
    msgs = state.get("messages", [])
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def fast_router(state: ClaimState) -> ClaimState:
    """
    Fast keyword-based router to detect claims and their type.
    Supports EN/FR/AR keywords.
    """
    # If we already have a claim in progress (followup), skip routing
    if state.get("claim_type") and state.get("claim_type") != "unknown":
        print(f"[ROUTER] Continuing existing {state.get('claim_type')} claim")
        return {
            "intent": "claim",
            "claim_type": state.get("claim_type"),
            "needs_llm_router": False,
            "router_reason": "Continuing existing claim conversation."
        }

    text = _norm(_latest_user_text(state))

    # Claim indicators (EN/FR/AR)
    claim_kw = [
        "claim", "file a claim", "insurance claim", "report a claim",
        "sinistre", "declaration", "declarer", "reclamation",
        "مطالبة", "تعويض", "تصريح", "تصريح بالحادث", "بلاغ"
    ]

    # Auto (EN/FR/AR)
    auto_kw = [
        "car", "vehicle", "auto", "accident", "crash", "collision", "plate",
        "voiture", "vehicule", "collision", "plaque",
        "سيارة", "عربة", "مركبة", "حادث", "اصطدام", "تصادم", "لوحة"
    ]

    # Home (EN/FR/AR)
    home_kw = [
        "house", "home", "apartment", "water leak", "leak", "fire", "burglary", "theft",
        "maison", "domicile", "appartement", "degat des eaux", "fuite", "incendie", "cambriolage", "vol",
        "منزل", "بيت", "شقة", "تسرب", "تسرب مياه", "حريق", "سرقة", "اقتحام"
    ]

    # Health (EN/FR/AR)
    health_kw = [
        "health", "medical", "doctor", "hospital", "surgery", "illness", "injury",
        "sante", "medical", "medecin", "hopital", "chirurgie", "maladie", "blessure",
        "صحة", "طبي", "طبيب", "مستشفى", "جراحة", "مرض", "إصابة"
    ]

    # Travel (EN/FR/AR)
    travel_kw = [
        "travel", "trip", "flight", "luggage", "baggage", "delay", "cancellation",
        "voyage", "vol", "bagage", "retard", "annulation",
        "سفر", "رحلة", "طيران", "أمتعة", "تأخير", "إلغاء"
    ]

    def hits(keywords: list[str]) -> int:
        return sum(1 for k in keywords if k in text)

    claim_hits = hits(claim_kw)
    auto_hits = hits(auto_kw)
    home_hits = hits(home_kw)
    health_hits = hits(health_kw)
    travel_hits = hits(travel_kw)

    # Not a claim
    if claim_hits == 0 and auto_hits == 0 and home_hits == 0 and health_hits == 0 and travel_hits == 0:
        return {
            "intent": "not_claim",
            "needs_llm_router": False,
            "router_reason": "No claim indicators detected.",
            "messages": [AIMessage(content="I can help you. Are you trying to file an insurance claim?")]
        }

    # Find best match
    scores = {
        "auto": auto_hits,
        "home": home_hits,
        "health": health_hits,
        "travel": travel_hits,
    }
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Strong signal
    if best_score >= 2:
        return {
            "intent": "claim",
            "claim_type": best_type,
            "needs_llm_router": False,
            "router_reason": f"Detected {best_type} claim keywords."
        }

    # Weak signal - still a claim but type unclear
    return {
        "intent": "claim",
        "claim_type": "unknown",
        "needs_llm_router": True,
        "router_reason": "Claim detected but type unclear."
    }


def route_after_fast(state: ClaimState) -> Literal["end_chat", "extract_fields"]:
    """Route after fast router: extract if claim, end if not."""
    if state.get("intent") == "not_claim":
        return "end_chat"
    return "extract_fields"
