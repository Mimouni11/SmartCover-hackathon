from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import graph
from langchain_core.messages import HumanMessage
import json
import os
from datetime import datetime

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLAIMS_FILE = "claims_storage.json"

# Initialize claims file if doesn't exist
if not os.path.exists(CLAIMS_FILE):
    with open(CLAIMS_FILE, 'w') as f:
        json.dump({"claims": []}, f)

def save_claim_to_storage(claim_data):
    """Save completed claim to JSON file for insurer dashboard"""
    try:
        with open(CLAIMS_FILE, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted, start fresh
        data = {"claims": []}
    
    # Add new claim
    data["claims"].append(claim_data)
    
    with open(CLAIMS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

class UserMessage(BaseModel):
    user_id: str
    message: str
    thread_id: str

@app.post("/chat")
def chat(msg: UserMessage):
    # Run your graph
    result = graph.invoke(
        {"messages": [HumanMessage(content=msg.message)]},
        config={"configurable": {"thread_id": msg.thread_id}}
    )
    
    # Get AI response
    ai_reply = result["messages"][-1].content
    
    # Check if claim is complete
    is_complete = result.get("predicted_cost") is not None
    
    if is_complete:
        # Extract structured data for description
        structured = result.get("structured_claims", {})
        claim_type = result.get("claim_type", "unknown")
        claim_fields = structured.get(claim_type, {})
        
        # Create a readable description
        description_parts = []
        for key, value in claim_fields.items():
            if value is not None and value != "":
                description_parts.append(f"{key}: {value}")
        description = " | ".join(description_parts[:5])  # First 5 fields
        
        # Prepare claim for insurer dashboard
        claim_for_insurer = {
            "id": f"CLM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "user_id": msg.user_id,
            "thread_id": msg.thread_id,
            "type": claim_type.capitalize(),
            "company": "BH Assurance",
            "amount": float(result.get("predicted_cost", 0)),
            "status": "pending",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "description": description,  # String, not dict!
            "fraudScore": float(result.get("fraud_score", 0)),
            "fraud_risk_level": result.get("fraud_risk_level", "LOW"),
            "fraud_signals": result.get("fraud_signals", []),
            "fraud_decision": result.get("fraud_decision", "AUTO_APPROVE"),
            "acceptance_probability": float(result.get("acceptance_probability", 0)),
            # Store structured data separately (JSON serializable)
            "structured_data": structured
        }
        
        # Save to JSON file
        save_claim_to_storage(claim_for_insurer)
        
        # Return ONLY acceptance to user
        return {
            "reply": f"✅ Votre réclamation a été soumise avec succès!\n\nProbabilité d'acceptation: {result.get('acceptance_probability', 0)*100:.0f}%\n\nVotre dossier est en cours de traitement.",
            "is_complete": True,
            "acceptance_probability": result.get("acceptance_probability", 0),
            "claim_id": claim_for_insurer["id"]
        }
    
    return {
        "reply": ai_reply,
        "is_complete": False
    }

@app.get("/claims")
def get_claims():
    """Endpoint for insurer dashboard to fetch all claims"""
    try:
        with open(CLAIMS_FILE, 'r') as f:
            data = json.load(f)
        return data["claims"]
    except (json.JSONDecodeError, FileNotFoundError):
        return []

@app.post("/claims/{claim_id}/update")
def update_claim_status(claim_id: str, status: str):
    """Update claim status (approved/rejected)"""
    try:
        with open(CLAIMS_FILE, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"success": False, "error": "No claims found"}
    
    for claim in data["claims"]:
        if claim["id"] == claim_id:
            claim["status"] = status
            break
    
    with open(CLAIMS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return {"success": True}