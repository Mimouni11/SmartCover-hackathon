"""
Extraction Node - Wraps llm_node to extract structured claim fields.
Uses Ollama/Mistral for extraction.
"""

from typing_extensions import Literal
from langchain_core.messages import HumanMessage, AIMessage

from .state import ClaimState
from llm_node import structure_claim


def _latest_user_text(state: ClaimState) -> str:
    """Get the last user message from state."""
    msgs = state.get("messages", [])
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def extract_claim_fields(state: ClaimState) -> ClaimState:
    """
    Call llm_node to extract structured fields from user message.
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
        for fields in state.get("missing_required_fields", {}).values():
            llm_input["missing_fields"].extend(fields)

    # Call llm_node's structure_claim function
    llm_result = structure_claim(llm_input)

    # DEBUG: Print full LLM result
    print(f"\n[EXTRACTION NODE - LLM RESULT]")
    print(f"  insurance_types: {llm_result.get('insurance_types', [])}")
    print(f"  structured_claims: {llm_result.get('structured_claims', {})}")
    print(f"  missing_required_fields: {llm_result.get('missing_required_fields', {})}")
    print(f"  confidence: {llm_result.get('confidence', 0)}")
    print(f"  needs_followup: {llm_result.get('needs_followup', False)}")
    print(f"  followup_question: {llm_result.get('followup_question', '')}")

    # Get detected type
    detected_types = llm_result.get("insurance_types", [])
    claim_type = detected_types[0] if detected_types else state.get("claim_type", "unknown")

    # Build response message
    if llm_result.get("needs_followup") and llm_result.get("followup_question"):
        reply = llm_result["followup_question"]
    else:
        reply = f"Got it! Claim type: {claim_type}. Processing your claim..."

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


def route_after_extraction(state: ClaimState) -> Literal["ask_followup", "run_cost_model"]:
    """Route after extraction: ask followup if missing fields, else run models."""
    needs_followup = state.get("needs_followup", False)
    missing = state.get("missing_required_fields", {})

    # Check if any required fields are missing
    has_missing = any(len(fields) > 0 for fields in missing.values())

    if needs_followup and has_missing:
        return "ask_followup"
    return "run_cost_model"
