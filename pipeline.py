# final_workflow.py
# Builds the FINAL LangGraph workflow by composing:
#   Agent 1 (intent + claim-type detection) -> Agent 2 (field collection) -> Agent 3 (model predictions)
#
# Assumes you already created these files:
#   agent1.py  -> exposes: agent1_graph
#   agent2.py  -> exposes: agent2_graph
#   agent3.py  -> exposes: agent3_graph
#
# Note: if Agent 2 uses interrupt()/resume later, compile the parent graph with a checkpointer. [web:162][web:47]

from typing import Any
from typing_extensions import TypedDict, Literal, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from agent1 import agent1_graph
from agent2 import agent2_graph
from agent3 import agent3_graph

Intent = Literal["claim", "not_claim"]
ClaimType = Literal["auto", "home", "life", "general", "unknown"]

class FinalState(TypedDict, total=False):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Outputs of agent 1
    intent: Intent
    claim_type: ClaimType

    # Inputs/outputs of agent 2
    user_text: str
    fields: dict[str, Any]
    pending_field: str

    # Outputs of agent 3
    predicted_cost: float
    fraud_score: float


# Optional: create a small “handoff” normalizer between agents (left empty per request)
def normalize_for_agent2(state: FinalState) -> FinalState:
    """
    Ensures fields exist before agent2 starts.
    Keep empty / implement later if needed.
    """
    raise NotImplementedError


def normalize_for_agent3(state: FinalState) -> FinalState:
    """
    Ensures agent3 gets claim_type + fields.
    Keep empty / implement later if needed.
    """
    raise NotImplementedError


builder = StateGraph(FinalState)

# Mount the sub-agents as nodes (graph-as-node composition). [web:162]
builder.add_node("agent1", agent1_graph)
builder.add_node("normalize_for_agent2", normalize_for_agent2)
builder.add_node("agent2", agent2_graph)
builder.add_node("normalize_for_agent3", normalize_for_agent3)
builder.add_node("agent3", agent3_graph)

# Sequential pipeline
builder.add_edge(START, "agent1")
builder.add_edge("agent1", "normalize_for_agent2")
builder.add_edge("normalize_for_agent2", "agent2")
builder.add_edge("agent2", "normalize_for_agent3")
builder.add_edge("normalize_for_agent3", "agent3")
builder.add_edge("agent3", END)

# Compile with checkpointer so conversations / interrupts can resume via thread_id. [web:47][web:162]
checkpointer = MemorySaver()
final_agent = builder.compile(checkpointer=checkpointer)

# Example invocation (you’ll pass config={"configurable":{"thread_id":"..."} } in your app):
# result = final_agent.invoke({"messages": [HumanMessage(content="...")]}, config={"configurable": {"thread_id": "u1"}})