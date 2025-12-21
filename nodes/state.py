"""Shared state definition for all nodes."""

from typing import Any
from typing_extensions import TypedDict, Literal, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


Intent = Literal["claim", "not_claim"]
ClaimType = Literal["auto", "home", "health", "travel", "life", "general", "other", "unknown"]


class ClaimState(TypedDict, total=False):
    # Conversation history
    messages: Annotated[list[BaseMessage], add_messages]

    # Router outputs
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

    # Model prediction outputs
    predicted_cost: float
    fraud_score: float
    acceptance_probability: float
    model_meta: dict[str, Any]
