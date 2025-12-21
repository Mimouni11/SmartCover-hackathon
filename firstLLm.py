import unicodedata
from typing import Any

from typing_extensions import TypedDict, Literal, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import your llm_node (uses Ollama/Mistral - no API key needed)
from llm_node import structure_claim


# ---- Types ----
ClaimType = Literal["auto", "home", "health", "travel", "life", "general", "other", "unknown"]
Intent = Literal["claim", "not_claim"]


class RouteState(TypedDict, total=False):
    # Conversation history (user + assistant)
    messages: Annotated[list[BaseMessage], add_messages]

    # Routing outputs
    intent: Intent
    claim_type: ClaimType
    needs_llm_router: bool
    router_reason: str

    # LLM extraction outputs (from llm_node)
    insurance_types: list[str]
    structured_claims: dict[str, Any]
    missing_required_fields: dict[str, list[str]]
    confidence: float
    needs_followup: bool
    followup_question: str
    extraction_status: str


def _norm(s: str) -> str:
    """Normalize text: lowercase + remove accents."""
    s = s.lower()
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    return s


def _latest_user_text(state: RouteState) -> str:
    msgs = state.get("messages", [])
    # Find last HumanMessage (safe even if graph adds AIMessage in between)
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


# ---- Node 1: heuristic router ----
def fast_router(state: RouteState) -> RouteState:
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
        "voiture", "vehicule", "accident", "collision", "plaque",
        "سيارة", "عربة", "مركبة", "حادث", "اصطدام", "تصادم", "لوحة"
    ]

    # Home (EN/FR/AR)
    home_kw = [
        "house", "home", "apartment", "water leak", "leak", "fire", "burglary", "theft",
        "maison", "domicile", "appartement", "degat des eaux", "fuite", "incendie", "cambriolage", "vol",
        "منزل", "بيت", "شقة", "تسرب", "تسرب مياه", "حريق", "سرقة", "اقتحام"
    ]

    # Life (EN/FR/AR) — adjust keywords later to match your dataset
    life_kw = [
        "life insurance", "death", "deceased", "beneficiary",
        "assurance vie", "deces", "beneficiaire",
        "تأمين على الحياة", "وفاة", "متوفى", "مستفيد"
    ]

    def hits(keywords: list[str]) -> int:
        return sum(1 for k in keywords if k in text)

    claim_hits = hits(claim_kw)
    auto_hits = hits(auto_kw)
    home_hits = hits(home_kw)
    life_hits = hits(life_kw)

    # Not a claim
    if claim_hits == 0 and auto_hits == 0 and home_hits == 0 and life_hits == 0:
        return {
            "intent": "not_claim",
            "needs_llm_router": False,
            "router_reason": "No claim indicators (EN/FR/AR).",
        }

    # Strong type signals
    if auto_hits >= 2 and auto_hits > max(home_hits, life_hits):
        return {"intent": "claim", "claim_type": "auto", "needs_llm_router": False, "router_reason": "Auto keywords."}

    if home_hits >= 2 and home_hits > max(auto_hits, life_hits):
        return {"intent": "claim", "claim_type": "home", "needs_llm_router": False, "router_reason": "Home keywords."}

    if life_hits >= 2 and life_hits > max(auto_hits, home_hits):
        return {"intent": "claim", "claim_type": "life", "needs_llm_router": False, "router_reason": "Life keywords."}

    # Claim-like but unclear => call LLM router later
    return {"intent": "claim", "claim_type": "unknown", "needs_llm_router": True, "router_reason": "Claim-like but unclear."}


# ---- Node 2: LLM router (stub) ----
def llm_router(state: RouteState) -> RouteState:
    # TODO: replace with structured LLM classifier (EN/FR/AR).
    # Temporary rule: unknown => general (your definition: not auto/home/life)
    if state.get("intent") == "claim" and state.get("claim_type") == "unknown":
        return {"claim_type": "general", "needs_llm_router": False, "router_reason": "Stub: default to general."}
    return {"needs_llm_router": False, "router_reason": "Stub: keep fast_router decision."}


# ---- Next-step placeholders ----
def normal_chat(state: RouteState) -> RouteState:
    """Handle non-claim conversations."""
    reply = "I can help. Are you trying to file an insurance claim?"
    return {"messages": [AIMessage(content=reply)]}


# ---- Node 3: Extract claim fields using llm_node ----
def extract_claim_fields(state: RouteState) -> RouteState:
    """
    Call the llm_node to extract structured fields from user message.
    Bridges LangGraph state <-> llm_node state.
    """
    user_text = _latest_user_text(state)

    # Prepare input for llm_node
    llm_input = {
        "user_input": user_text,
        "user_id": "langgraph_user",
        "is_followup": state.get("needs_followup", False),
    }

    # If this is a followup, include previous data
    if state.get("needs_followup"):
        llm_input["previous_claim_data"] = state.get("structured_claims", {})
        llm_input["missing_fields"] = []
        # Flatten missing fields from all types
        for fields in state.get("missing_required_fields", {}).values():
            llm_input["missing_fields"].extend(fields)

    # Call llm_node's structure_claim function
    llm_result = structure_claim(llm_input)

    # Map claim_type to first detected type if available
    detected_types = llm_result.get("insurance_types", [])
    claim_type = detected_types[0] if detected_types else state.get("claim_type", "unknown")

    # Build response message based on extraction result
    if llm_result.get("needs_followup") and llm_result.get("followup_question"):
        reply = llm_result["followup_question"]
    else:
        reply = f"Claim extracted. Type: {claim_type}. Confidence: {llm_result.get('confidence', 0):.0%}"

    return {
        "claim_type": claim_type,
        "insurance_types": detected_types,
        "structured_claims": llm_result.get("structured_claims", {}),
        "missing_required_fields": llm_result.get("missing_required_fields", {}),
        "confidence": llm_result.get("confidence", 0.0),
        "needs_followup": llm_result.get("needs_followup", True),
        "followup_question": llm_result.get("followup_question", ""),
        "extraction_status": llm_result.get("extraction_status", "unknown"),
        "messages": [AIMessage(content=reply)],
    }


# ---- Node 4: Finalize claim (all required fields collected) ----
def finalize_claim(state: RouteState) -> RouteState:
    """Called when all required fields are collected."""
    claim_type = state.get("claim_type", "unknown")
    structured = state.get("structured_claims", {})
    confidence = state.get("confidence", 0.0)

    reply = (
        f"Your {claim_type} claim has been recorded.\n"
        f"Confidence: {confidence:.0%}\n"
        f"Fields collected: {list(structured.get(claim_type, {}).keys())}"
    )
    return {"messages": [AIMessage(content=reply)]}


# ---- Conditional routing ----
def route_after_fast(state: RouteState) -> Literal["normal_chat", "llm_router", "extract_fields"]:
    """Route after fast keyword router."""
    if state.get("intent") == "not_claim":
        return "normal_chat"
    if state.get("needs_llm_router"):
        return "llm_router"
    # Claim detected with known type -> extract fields
    return "extract_fields"


def route_after_llm(state: RouteState) -> Literal["normal_chat", "extract_fields"]:
    """Route after LLM classifier."""
    return "extract_fields" if state.get("intent") == "claim" else "normal_chat"


def route_after_extraction(state: RouteState) -> Literal["extract_fields", "finalize_claim"]:
    """Route after extraction: continue if missing fields, finalize if complete."""
    if state.get("needs_followup") and state.get("missing_required_fields"):
        # Still missing required fields -> loop back for more input
        return "extract_fields"
    return "finalize_claim"


# ---- Build graph ----
builder = StateGraph(RouteState)

# Add all nodes
builder.add_node("fast_router", fast_router)
builder.add_node("llm_router", llm_router)
builder.add_node("normal_chat", normal_chat)
builder.add_node("extract_fields", extract_claim_fields)  # YOUR llm_node integration
builder.add_node("finalize_claim", finalize_claim)

# Define flow
builder.add_edge(START, "fast_router")
builder.add_conditional_edges(
    "fast_router",
    route_after_fast,
    ["normal_chat", "llm_router", "extract_fields"]
)
builder.add_conditional_edges(
    "llm_router",
    route_after_llm,
    ["normal_chat", "extract_fields"]
)
builder.add_conditional_edges(
    "extract_fields",
    route_after_extraction,
    ["extract_fields", "finalize_claim"]
)

# Terminal nodes
builder.add_edge("normal_chat", END)
builder.add_edge("finalize_claim", END)

graph = builder.compile()


# ---- Run the graph ----
if __name__ == "__main__":
    # Test 1: Non-claim message
    print("=== Test 1: Non-claim ===")
    out = graph.invoke({"messages": [HumanMessage(content="salut what is new for today.")]})
    print(f"Intent: {out.get('intent')}, Type: {out.get('claim_type')}")
    print(f"Last AI: {out['messages'][-1].content}\n")

    # Test 2: Auto claim
    print("=== Test 2: Auto claim ===")
    out = graph.invoke({
        "messages": [HumanMessage(content="J'ai eu un accident de voiture hier. Un camion m'a percuté.")]
    })
    print(f"Intent: {out.get('intent')}, Type: {out.get('claim_type')}")
    print(f"Confidence: {out.get('confidence', 0):.0%}")
    print(f"Structured: {out.get('structured_claims', {})}")
    print(f"Missing: {out.get('missing_required_fields', {})}")
    print(f"Last AI: {out['messages'][-1].content}\n")

    # Test 3: Home claim
    print("=== Test 3: Home claim ===")
    out = graph.invoke({
        "messages": [HumanMessage(content="Il y a une fuite d'eau dans ma maison depuis ce matin.")]
    })
    print(f"Intent: {out.get('intent')}, Type: {out.get('claim_type')}")
    print(f"Confidence: {out.get('confidence', 0):.0%}")
    print(f"Last AI: {out['messages'][-1].content}")
