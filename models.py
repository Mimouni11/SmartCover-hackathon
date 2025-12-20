from typing_extensions import TypedDict, Literal
from typing import Any


Intent = Literal["claim", "not_claim"]
ClaimType = Literal["auto", "home", "general", "other", "unknown"]

class ClaimState(TypedDict, total=False):
    user_text: str
    intent: Intent
    claim_type: ClaimType
    fields: dict[str, Any]

    # Step 3 outputs:
    predicted_cost: float
    fraud_score: float
    acceptance_probability: float  # third model output (stub)

    # Optional: keep the raw model outputs
    model_meta: dict[str, Any]

def cost_model_predict(fields: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    # TODO: replace with your real cost model
    return 1234.56, {"model": "cost_stub_v1"}

def fraud_model_predict(fields: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    # TODO: replace with your real fraud model
    return 0.12, {"model": "fraud_stub_v1"}

def acceptance_model_predict(fields: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    # TODO: replace with your third model (approval probability)
    return 0.78, {"model": "acceptance_stub_v1"}



from langgraph.graph import StateGraph, START, END

# --- Three model nodes ---
def run_cost_model(state: ClaimState) -> ClaimState:
    fields = state.get("fields") or {}
    cost, meta = cost_model_predict(fields)
    model_meta = state.get("model_meta") or {}
    model_meta["cost"] = meta
    return {"predicted_cost": cost, "model_meta": model_meta}

def run_fraud_model(state: ClaimState) -> ClaimState:
    fields = state.get("fields") or {}
    fraud, meta = fraud_model_predict(fields)
    model_meta = state.get("model_meta") or {}
    model_meta["fraud"] = meta
    return {"fraud_score": fraud, "model_meta": model_meta}

def run_acceptance_model(state: ClaimState) -> ClaimState:
    fields = state.get("fields") or {}
    p, meta = acceptance_model_predict(fields)
    model_meta = state.get("model_meta") or {}
    model_meta["acceptance"] = meta
    return {"acceptance_probability": p, "model_meta": model_meta}

# --- Join node (fan-in) ---
def models_done(state: ClaimState) -> ClaimState:
    # Step 4 will generate the user message from these three outputs
    # (No extra computation yet)
    return state


builder = StateGraph(ClaimState)

builder.add_node("run_cost_model", run_cost_model)
builder.add_node("run_fraud_model", run_fraud_model)
builder.add_node("run_acceptance_model", run_acceptance_model)
builder.add_node("models_done", models_done)

# Entry for step 3 (you will connect step 2 "ready_for_models" -> these)
builder.add_edge(START, "run_cost_model")
builder.add_edge(START, "run_fraud_model")
builder.add_edge(START, "run_acceptance_model")

# Fan-in join: run models_done after all three finish
builder.add_edge(["run_cost_model", "run_fraud_model", "run_acceptance_model"], "models_done")

builder.add_edge("models_done", END)

models_graph = builder.compile()
