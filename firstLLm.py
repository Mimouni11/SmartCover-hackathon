import os
import unicodedata
from typing import Any
from typing_extensions import TypedDict, Literal, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ---- Config ----
LLM_API_KEY = os.getenv("LLM_API_KEY", "AIzaSyAB8Ob9N6F3saAPM5zig286S41ZcLqbuIE")  # do NOT hardcode real keys


# ---- Types ----
ClaimType = Literal["auto", "home", "life", "general", "other", "unknown"]
Intent = Literal["claim", "not_claim"]

class RouteState(TypedDict, total=False):
    # Conversation history (user + assistant)
    messages: Annotated[list[BaseMessage], add_messages]  # appends automatically [web:7][web:141]

    # Routing outputs
    intent: Intent
    claim_type: ClaimType
    needs_llm_router: bool
    router_reason: str


# def _norm(s: str) -> str:
#     s = s.lower()
#     s = "".join(
#         c for c in unicodedata.normalize("NFKD", s)
#         if not unicodedata.combining(c)
#     )
#     return s

def claim_detection(state: RouteState) -> RouteState:
    # text = _norm(_latest_user_text(state))
    # claim_kw = [
    #     "claim", "file a claim", "insurance claim", "report a claim",
    #     "sinistre", "declaration", "declarer", "reclamation",
    #     "مطالبة", "تعويض", "تصريح", "تصريح بالحادث", "بلاغ"
    # ]
    # if any(kw in text for kw in claim_kw):
    #     return {"intent": "claim"}
    # return {"intent": "not_claim"}
def detect_claim_type(state: RouteState) -> RouteState:
    # text = _norm(_latest_user_text(state))
    # claim_type_kw = {
    #     "auto": [
    #         "car", "vehicle", "auto", "accident", "crash", "collision", "plate",
    #         "voiture", "vehicule", "accident", "collision", "plaque",
    #         "سيارة", "عربة", "مركبة", "حادث", "اصطدام", "تصادم", "لوحة"
    #     ],
    #     "home": [
    #         "house", "home", "apartment", "water leak", "leak", "fire", "burglary", "theft",
    #         "maison", "domicile", "appartement", "degat des eaux", "fuite", "incendie", "cambriolage", "vol",
    #         "منزل", "بيت", "شقة", "تسرب", "تسرب مياه", "حريق", "سرقة", "اقتحام"
    #     ],
    #     "life": [
    #         "life insurance", "death", "deceased", "beneficiary",
    #         "assurance vie", "deces", "beneficiaire",
    #         "تأمين على الحياة", "وفاة", "متوفى", "مستفيد"
    #     ]
    # }
    # for claim_type, keywords in claim_type_kw.items():
    #     if any(kw in text for kw in keywords):
    #         return {"claim_type": claim_type}
    # return {"claim_type": "unknown"}
def 
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
    # Stub response + record it in history
    reply = "I can help. Are you trying to file an insurance claim?"
    return {"messages": [AIMessage(content=reply)]}

def claim_flow_entry(state: RouteState) -> RouteState:
    # This is where Step 2 starts later (collect required fields)
    reply = f"Claim detected. Type: {state.get('claim_type', 'unknown')}."
    return {"messages": [AIMessage(content=reply)]}


# ---- Conditional routing ----
def route_after_fast(state: RouteState) -> Literal["normal_chat", "llm_router", "claim_flow_entry"]:
    if state.get("intent") == "not_claim":
        return "normal_chat"
    if state.get("needs_llm_router"):
        return "llm_router"
    return "claim_flow_entry"

def route_after_llm(state: RouteState) -> Literal["normal_chat", "claim_flow_entry"]:
    return "claim_flow_entry" if state.get("intent") == "claim" else "normal_chat"


# ---- Build graph ----
builder = StateGraph(RouteState)

builder.add_node("fast_router", fast_router)
builder.add_node("llm_router", llm_router)
builder.add_node("normal_chat", normal_chat)
builder.add_node("claim_flow_entry", claim_flow_entry)

builder.add_edge(START, "fast_router")
builder.add_conditional_edges("fast_router", route_after_fast, ["normal_chat", "llm_router", "claim_flow_entry"])
builder.add_conditional_edges("llm_router", route_after_llm, ["normal_chat", "claim_flow_entry"])

builder.add_edge("normal_chat", END)
builder.add_edge("claim_flow_entry", END)

graph = builder.compile()

# Example usage: pass the user's message as a HumanMessage so it lands in history
out = graph.invoke({"messages": [HumanMessage(content="salut what is new for today.")]})
print(out["intent"], out.get("claim_type"))
print("History size:", len(out["messages"]))
print("Last AI:", out["messages"][-1].content)
