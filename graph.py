"""
Main LangGraph Pipeline - Orchestrates all nodes.

Flow:
    START -> fast_router -> extract_fields -> [followup loop] -> run_models -> finalize -> END
                        |
                        v (if not a claim)
                      end_chat -> END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import state and all nodes
from nodes.state import ClaimState
from nodes.router_node import fast_router, route_after_fast
from nodes.extraction_node import extract_claim_fields, route_after_extraction
from nodes.models_node import (
    run_cost_model,
    run_fraud_model,
    run_acceptance_model,
    finalize_claim,
)
from langchain_core.messages import AIMessage, HumanMessage


def end_chat(state: ClaimState) -> ClaimState:
    """End node for non-claim conversations."""
    return state  # Message already added by router


def ask_followup(state: ClaimState) -> ClaimState:
    """Node to ask for missing information."""
    question = state.get("followup_question", "Can you provide more details?")
    return {"messages": [AIMessage(content=question)]}


# ============ BUILD THE GRAPH ============

builder = StateGraph(ClaimState)

# Add all nodes
builder.add_node("fast_router", fast_router)
builder.add_node("end_chat", end_chat)
builder.add_node("extract_fields", extract_claim_fields)
builder.add_node("ask_followup", ask_followup)
builder.add_node("run_cost_model", run_cost_model)
builder.add_node("run_fraud_model", run_fraud_model)
builder.add_node("run_acceptance_model", run_acceptance_model)
builder.add_node("finalize_claim", finalize_claim)

# Define the flow
builder.add_edge(START, "fast_router")

# After router: either end chat or extract fields
builder.add_conditional_edges(
    "fast_router",
    route_after_fast,
    ["end_chat", "extract_fields"]
)

builder.add_edge("end_chat", END)

# After extraction: either ask followup or run models
builder.add_conditional_edges(
    "extract_fields",
    route_after_extraction,
    ["ask_followup", "run_cost_model"]  # run_cost_model starts the model chain
)

# Followup goes back to extraction (would need user input in real app)
builder.add_edge("ask_followup", END)  # For now, end here - interactive loop handles it

# Run models sequentially: cost -> fraud -> acceptance -> finalize
builder.add_edge("run_cost_model", "run_fraud_model")
builder.add_edge("run_fraud_model", "run_acceptance_model")
builder.add_edge("run_acceptance_model", "finalize_claim")

builder.add_edge("finalize_claim", END)

# Compile the graph with memory for conversation tracking
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ============ INTERACTIVE CONSOLE ============

def run_interactive():
    """Run the pipeline interactively from console with conversation memory."""
    import uuid

    print("\n" + "=" * 60)
    print("  INSURANCE CLAIM PROCESSOR")
    print("  Describe your situation (EN/FR/AR supported)")
    print("  Type 'quit' to exit, 'new' for new conversation")
    print("=" * 60 + "\n")

    # Create a unique thread ID for this conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print("\n--- New conversation started ---\n")
            continue

        # Run the graph with conversation memory
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        # Print the last AI message
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            print(f"\nA: {last_msg.content}\n")

        # If claim was processed, show summary
        if result.get("predicted_cost"):
            print(f"  Cost: ${result['predicted_cost']:,.2f}")
            print(f"  Fraud Risk: {result.get('fraud_score', 0):.1%}")
            print(f"  Acceptance: {result.get('acceptance_probability', 0):.1%}")
            print()


if __name__ == "__main__":
    run_interactive()
