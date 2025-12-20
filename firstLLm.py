import os
import unicodedata
from typing_extensions import TypedDict, Literal

from langgraph.graph import StateGraph ,START,END
from typing_extensions import TypedDict, Literal

ClaimType = Literal["auto", "home", "general", "other", "unknown"]
Intent = Literal["claim", "not_claim"]

class RouteState(TypedDict, total=False):
    user_text: str
    intent: Intent                 # "claim" | "not_claim"
    claim_type: ClaimType          # only meaningful if intent == "claim"
    needs_llm_router: bool         # True when heuristics are unsure
    router_reason: str             # optional, useful for debugging


# --- API key placeholder (you will replace later) ---
LLM_API_KEY = os.getenv("LLM_API_KEY", "AIzaSyAB8Ob9N6F3saAPM5zig286S41ZcLqbuIE")
# Optionally set the provider-specific env var later, e.g.:
# os.environ["OPENAI_API_KEY"] = LLM_API_KEY


Intent = Literal["claim", "not_claim"]
ClaimType = Literal["auto", "home", "general", "other", "unknown"]

class RouteState(TypedDict, total=False):
    user_text: str
    intent: Intent
    claim_type: ClaimType
    needs_llm_router: bool
    router_reason: str


def _norm(s: str) -> str:
    """Lowercase + remove diacritics (helps French matching)."""
    s = s.lower()
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    return s


# --- Node 1: cheap heuristic router ---
def fast_router(state: RouteState) -> RouteState:
    text = _norm(state.get("user_text", ""))

    # Claim indicators (EN/FR/AR)
    claim_kw = [
        "claim", "file a claim", "insurance claim", "report a claim",
        "sinistre", "declaration", "declarer", "reclamation",
        "مطالبة", "تعويض", "تصريح", "تصريح بالحادث", "بلاغ"
    ]

    # Auto (EN/FR/AR)
    auto_kw = [
        "car", "vehicle", "auto", "accident", "crash", "collision", "plate",
        "voiture", "vehicule", "auto", "accident", "collision", "plaque",
        "سيارة", "عربة", "مركبة", "حادث", "اصطدام", "تصادم", "لوحة"
    ]

    # Home (EN/FR/AR)
    home_kw = [
        "house", "home", "apartment", "water leak", "leak", "fire", "burglary", "theft",
        "maison", "domicile", "appartement", "degat des eaux", "fuite", "incendie", "cambriolage", "vol",
        "منزل", "بيت", "شقة", "تسرب", "تسرب مياه", "حريق", "سرقة", "اقتحام"
    ]

    def hits(keywords: list[str]) -> int:
        return sum(1 for k in keywords if k in text)

    claim_hits = hits(claim_kw)
    auto_hits = hits(auto_kw)
    home_hits = hits(home_kw)

    # Not a claim (cheap decision)
    if claim_hits == 0 and auto_hits == 0 and home_hits == 0:
        return {
            "intent": "not_claim",
            "needs_llm_router": False,
            "router_reason": "No claim indicators (EN/FR/AR).",
        }

    # Clearly auto or home
    if auto_hits >= 2 and auto_hits > home_hits:
        return {
            "intent": "claim",
            "claim_type": "auto",
            "needs_llm_router": False,
            "router_reason": "Auto keywords (EN/FR/AR).",
        }

    if home_hits >= 2 and home_hits > auto_hits:
        return {
            "intent": "claim",
            "claim_type": "home",
            "needs_llm_router": False,
            "router_reason": "Home keywords (EN/FR/AR).",
        }

    # It's claim-like, but type is unclear -> go to LLM router
    return {
        "intent": "claim",
        "claim_type": "unknown",
        "needs_llm_router": True,
        "router_reason": "Claim-like, but type unclear.",
    }


# --- Node 2: LLM router (stub for now) ---
def llm_router(state: RouteState) -> RouteState:
    """
    TODO (later): call LLM and return structured output:
      intent: claim|not_claim
      claim_type: auto|home|general|other
      confidence: 0..1
    """
    # Temporary default policy:
    # If fast_router said it's a claim but unknown type -> treat as general
    if state.get("intent") == "claim" and state.get("claim_type") == "unknown":
        return {"claim_type": "general", "needs_llm_router": False, "router_reason": "Stub: default to general."}
    return {"needs_llm_router": False, "router_reason": "Stub: keep fast_router decision."}


def normal_chat(state: RouteState) -> RouteState:
    # Placeholder for non-claim assistant behavior
    return state


def claim_flow_entry(state: RouteState) -> RouteState:
    # Placeholder: next step will be field collection based on claim_type
    return state


# --- Conditional routing functions ---
def route_after_fast(state: RouteState) -> Literal["normal_chat", "llm_router", "claim_flow_entry"]:
    if state.get("intent") == "not_claim":
        return "normal_chat"
    if state.get("needs_llm_router"):
        return "llm_router"
    return "claim_flow_entry"


def route_after_llm(state: RouteState) -> Literal["normal_chat", "claim_flow_entry"]:
    return "claim_flow_entry" if state.get("intent") == "claim" else "normal_chat"


# --- Build graph ---
builder = StateGraph(RouteState)

builder.add_node("fast_router", fast_router)
builder.add_node("llm_router", llm_router)
builder.add_node("normal_chat", normal_chat)
builder.add_node("claim_flow_entry", claim_flow_entry)

builder.add_edge(START, "fast_router")

# Provide allowed destinations (helps clarity/visualization and matches common patterns)
builder.add_conditional_edges("fast_router", route_after_fast, ["normal_chat", "llm_router", "claim_flow_entry"])
builder.add_conditional_edges("llm_router", route_after_llm, ["normal_chat", "claim_flow_entry"])

builder.add_edge("normal_chat", END)
builder.add_edge("claim_flow_entry", END)

graph = builder.compile()


# Example usage:
out = graph.invoke({"user_text": "J'ai eu un accident avec ma voiture hier."})
print(out)
