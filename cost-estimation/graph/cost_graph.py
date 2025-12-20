"""
LangGraph workflow pour l'estimation de coûts
"""
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from cost.agent import cost_estimation_agent


# ----- State -----
class CostState(TypedDict):
    anonymized_claim: dict
    cost_analysis: dict


# ----- Node -----
def cost_estimation_node(state: CostState) -> dict:
    """
    Nœud d'estimation de coûts
    """
    claim = state.get("anonymized_claim")
    if claim is None:
        raise ValueError("La clé 'anonymized_claim' est manquante dans l'état !")
    
    # Appeler l'agent d'estimation
    result = cost_estimation_agent(claim)
    
    return {"cost_analysis": result}


# ----- Graph -----
def build_cost_graph():
    """
    Construire le workflow LangGraph
    """
    graph = StateGraph(CostState)
    graph.add_node("cost_estimation", cost_estimation_node)
    graph.set_entry_point("cost_estimation")
    graph.add_edge("cost_estimation", END)
    return graph.compile()
