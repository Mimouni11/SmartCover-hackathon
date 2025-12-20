from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict  # ✅ Import correct
from fraud.agent import fraud_agent


# ----- State avec TypedDict (pas dict) -----
class ClaimState(TypedDict):
    anonymized_claim: dict
    fraud_analysis: dict


# ----- Node -----
def fraud_detection_node(state: ClaimState):
    print(f"🔍 État reçu: {state.keys()}")  # Debug
    
    claim = state.get("anonymized_claim")
    if claim is None:
        raise ValueError("La clé 'anonymized_claim' est manquante dans l'état !")
    
    # Appeler l'agent de détection de fraude
    result = fraud_agent(claim)
    
    # Retourner uniquement les nouvelles clés à ajouter
    return {"fraud_analysis": result}


# ----- Graph -----
def build_fraud_graph():
    graph = StateGraph(ClaimState)
    graph.add_node("fraud_detection", fraud_detection_node)
    graph.set_entry_point("fraud_detection")
    graph.add_edge("fraud_detection", END)
    return graph.compile()
