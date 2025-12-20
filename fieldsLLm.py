from langgraph.types import interrupt
from typing_extensions import TypedDict, Literal
from typing import Any

Intent = Literal["claim", "not_claim"]
ClaimType = Literal["auto", "home", "general", "other", "unknown"]

class ClaimState(TypedDict, total=False):
    user_text: str
    intent: Intent
    claim_type: ClaimType

    # Step 2 additions:
    fields: dict[str, Any]          # collected structured fields
    pending_field: str              # current field being requested

# --- YOU will implement this later ---
def get_required_fields(claim_type: str) -> list[str]:
    """
    Return the list of required fields for a given claim type.
    Examples later (YOU fill):
      auto -> ["policy_number", "incident_date", "location", ...]
      home -> ["policy_number", "address", "damage_type", ...]
      general -> [...]
    """
    raise NotImplementedError("Fill required fields per claim_type later.")


def _make_question(field_name: str, claim_type: str) -> str:
    # Keep it simple now; later you can add EN/FR/AR templates.
    return f"To process your {claim_type} claim, please provide: {field_name}"


def collect_missing_fields(state: ClaimState) -> ClaimState:
    claim_type = state.get("claim_type", "unknown")
    fields = state.get("fields") or {}

    required = get_required_fields(claim_type)  # <-- you will define later
    missing = [f for f in required if f not in fields or fields[f] in (None, "", [])]

    if not missing:
        # Done collecting
        return {"fields": fields, "pending_field": ""}

    # Ask for the next missing field
    pending = missing[0]
    question = _make_question(pending, claim_type)

    # Pause graph execution here until user responds, then resume with Command(resume=...)
    answer = interrupt(question)  # answer is provided on resume [web:12]

    # Save into state
    fields[pending] = answer
    return {"fields": fields, "pending_field": pending}


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Literal

def route_after_collect(state: ClaimState) -> Literal["collect_missing_fields", "ready_for_models"]:
    claim_type = state.get("claim_type", "unknown")
    fields = state.get("fields") or {}

    required = get_required_fields(claim_type)  # you will define later
    missing = [f for f in required if f not in fields or fields[f] in (None, "", [])]

    return "collect_missing_fields" if missing else "ready_for_models"


def ready_for_models(state: ClaimState) -> ClaimState:
    # Step 3 will start from here (cost/fraud/etc.)
    return state


builder = StateGraph(ClaimState)

builder.add_node("collect_missing_fields", collect_missing_fields)
builder.add_node("ready_for_models", ready_for_models)

builder.add_edge(START, "collect_missing_fields")
builder.add_conditional_edges(
    "collect_missing_fields",
    route_after_collect,
    ["collect_missing_fields", "ready_for_models"],
)
builder.add_edge("ready_for_models", END)

# Checkpointer is important for interrupts/resume workflows [web:12]
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


from langgraph.types import Command

config = {"configurable": {"thread_id": "demo-1"}}

# First run will likely interrupt asking for the first missing field:
graph.invoke({"claim_type": "auto", "fields": {}}, config=config)

# Resume with user answer:
graph.invoke(Command(resume="ABC-123"), config=config)
