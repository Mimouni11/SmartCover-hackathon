"""
Models Node - Cost estimation, fraud detection, and acceptance models.
Runs after extraction is complete.
"""

import pickle
import numpy as np
import json
import os
import sys
from typing import Any
from langchain_core.messages import AIMessage

from .state import ClaimState

# Add project root to path for fraud module import
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from fraud.agent import fraud_agent


# ============ LOAD MODELS ============

# Get the project root directory (parent of nodes/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

print(f"[DEBUG] Looking for models in: {os.path.abspath(MODELS_DIR)}")

# Load pickle models
# Note: fraud detection is handled by fraud/agent.py which loads its own models
cost_model_health = None

try:
    cost_model_path = os.path.join(MODELS_DIR, "cost_estimation_health_insurance.pkl")
    with open(cost_model_path, "rb") as f:
        cost_model_health = pickle.load(f)
    print(f"✓ Loaded cost_model_health from {cost_model_path}")
except FileNotFoundError as e:
    print(f"⚠️  cost_estimation_health_insurance.pkl not found at {cost_model_path}")
except Exception as e:
    print(f"⚠️  Error loading cost_model_health: {e}")


# ============ LOAD USER PROFILES DATABASE ============

USER_PROFILES_PATH = os.path.join(PROJECT_ROOT, "user_profiles.json")

try:
    with open(USER_PROFILES_PATH, "r") as f:
        USER_PROFILES = json.load(f)
    print(f"✓ Loaded {len(USER_PROFILES)} user profiles from user_profiles.json")
except FileNotFoundError:
    print(f"⚠️  user_profiles.json not found at {USER_PROFILES_PATH}")
    USER_PROFILES = {}


# ============ HELPER FUNCTIONS ============

def get_user_profile_data(user_id: str, claim_type: str) -> dict:
    """
    Fetch user profile from JSON database simulation.
    In production, this would query your actual database.
    """
    user_data = USER_PROFILES.get(user_id, {})
    profile = user_data.get(claim_type, {})
    
    if not profile:
        print(f"⚠️  No profile found for user={user_id}, claim_type={claim_type}")
        # Return minimal defaults
        if claim_type == "auto":
            return {
                "AGE": 30, "INCOME": 40000, "BLUEBOOK": 10000, "CAR_AGE": 5,
                "KIDSDRIV": 0, "HOMEKIDS": 0, "YOJ": 5, "HOME_VAL": 100000,
                "MSTATUS": "Single", "GENDER": "M", "EDUCATION": "Bachelors",
                "OCCUPATION": "Professional", "TRAVTIME": 30, "CAR_USE": "Private",
                "TIF": 3, "CAR_TYPE": "Sedan", "RED_CAR": "no", "OLDCLAIM": 0,
                "CLM_FREQ": 0, "REVOKED": "No", "MVR_PTS": 0, "URBANICITY": "Urban"
            }
        elif claim_type == "health":
            return {
                "age": 30, "sex": "male", "bmi": 25.0, "children": 0,
                "smoker": "no", "region": "northwest"
            }
        elif claim_type == "home":
            return {
                "house_age": 10, "house_size": 1500, "num_rooms": 3,
                "num_bathrooms": 2, "location": "urban", "building_material": "brick",
                "security_features": "none", "previous_claims": 0
            }
    
    return profile


def encode_categorical(value: str, mapping: dict) -> int:
    """Encode categorical variable to numeric."""
    return mapping.get(value, 0)


def prepare_auto_features(profile: dict) -> np.array:
    """
    Prepare features for auto insurance model.
    Order must match training data.
    """
    # Encode categorical variables
    mstatus_map = {"Single": 0, "Married": 1, "Divorced": 2}
    gender_map = {"M": 0, "F": 1}
    education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    occupation_map = {"Student": 0, "Professional": 1, "Blue Collar": 2, "Clerical": 3}
    car_use_map = {"Private": 0, "Commercial": 1}
    car_type_map = {"Sedan": 0, "SUV": 1, "Truck": 2, "Sports": 3, "Van": 4}
    red_car_map = {"no": 0, "yes": 1}
    revoked_map = {"No": 0, "Yes": 1}
    urbanicity_map = {"Urban": 0, "Highly Urban/ Urban": 1, "z_Highly Rural/ Rural": 2}
    
    # Build feature vector in correct order
    features = [
        profile.get("AGE", 30),
        profile.get("KIDSDRIV", 0),
        profile.get("YOJ", 5),
        profile.get("INCOME", 40000),
        profile.get("HOMEKIDS", 0),
        profile.get("HOME_VAL", 100000),
        encode_categorical(profile.get("MSTATUS", "Single"), mstatus_map),
        encode_categorical(profile.get("GENDER", "M"), gender_map),
        encode_categorical(profile.get("EDUCATION", "Bachelors"), education_map),
        encode_categorical(profile.get("OCCUPATION", "Professional"), occupation_map),
        profile.get("TRAVTIME", 30),
        encode_categorical(profile.get("CAR_USE", "Private"), car_use_map),
        profile.get("BLUEBOOK", 10000),
        profile.get("TIF", 3),
        encode_categorical(profile.get("CAR_TYPE", "Sedan"), car_type_map),
        encode_categorical(profile.get("RED_CAR", "no"), red_car_map),
        profile.get("OLDCLAIM", 0),
        profile.get("CLM_FREQ", 0),
        encode_categorical(profile.get("REVOKED", "No"), revoked_map),
        profile.get("MVR_PTS", 0),
        profile.get("CAR_AGE", 5),
        encode_categorical(profile.get("URBANICITY", "Urban"), urbanicity_map),
    ]
    
    return np.array(features).reshape(1, -1)


def prepare_health_features(profile: dict) -> np.array:
    """
    Prepare features for health insurance model.
    Region is ONE-HOT encoded.
    """
    sex_map = {"male": 1, "female": 0}
    smoker_map = {"yes": 1, "no": 0}

    region = profile.get("region", "northwest")

    # One-hot encode region (drop southwest as reference)
    region_northeast = 1 if region == "northeast" else 0
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0

    features = [
        profile.get("age", 30),
        encode_categorical(profile.get("sex", "male"), sex_map),
        profile.get("bmi", 25.0),
        profile.get("children", 0),
        encode_categorical(profile.get("smoker", "no"), smoker_map),
        region_northeast,
        region_northwest,
        region_southeast,
    ]

    return np.array(features).reshape(1, -1)


def extract_hour(time_str: str) -> int:
    """Extract hour from time string like '14:00'"""
    if not time_str:
        return 12  # default noon
    try:
        return int(str(time_str).split(":")[0])
    except:
        return 12


def prepare_claim_for_fraud_agent(extracted_fields: dict, profile: dict) -> dict:
    """
    Map LLM-extracted fields + profile to fraud model features.
    Required features by fraud/agent.py:
        months_as_customer, age, policy_deductable, policy_annual_premium,
        umbrella_limit, capital-gains, capital-loss, incident_hour_of_the_day,
        number_of_vehicles_involved, bodily_injuries, witnesses,
        total_claim_amount, injury_claim, property_claim, vehicle_claim
    """
    return {
        # From profile (customer data)
        "months_as_customer": profile.get("months_as_customer", 12),
        "age": profile.get("AGE", profile.get("age", 30)),
        "policy_deductable": profile.get("policy_deductable", 1000),
        "policy_annual_premium": profile.get("policy_annual_premium", 1200),
        "umbrella_limit": profile.get("umbrella_limit", 0),
        "capital-gains": profile.get("capital-gains", 0),
        "capital-loss": profile.get("capital-loss", 0),

        # From extracted fields (incident data)
        "incident_hour_of_the_day": extract_hour(extracted_fields.get("accident_time")),
        "number_of_vehicles_involved": 2 if extracted_fields.get("other_party_involved") else 1,
        "bodily_injuries": 1 if extracted_fields.get("injuries") else 0,
        "witnesses": 1 if extracted_fields.get("witness_present") else 0,
        "total_claim_amount": extracted_fields.get("total_amount", 0) or 0,

        # Claim breakdown (defaults if not provided)
        "injury_claim": extracted_fields.get("injury_claim", 0) or 0,
        "property_claim": extracted_fields.get("property_claim", 0) or 0,
        "vehicle_claim": extracted_fields.get("vehicle_claim", 0) or 0,
    }


# ============ MODEL PREDICTION FUNCTIONS ============

def cost_model_predict(claim_type: str, extracted_fields: dict, profile: dict) -> tuple[float, dict]:
    """
    Predict the cost of a claim using real ML models.
    """
    if claim_type == "health" and cost_model_health is not None:
        features = prepare_health_features(profile)
        cost = float(cost_model_health.predict(features)[0])
        return cost, {"model": "health_cost_ml_v1", "features_used": len(features[0])}
    
    elif claim_type == "auto":
        # Stub for auto (no trained model yet)
        base = 2500.0
        # Adjust based on extracted fields
        if extracted_fields.get("collision_type") == "rear":
            base *= 1.2
        if extracted_fields.get("injuries"):
            base *= 1.5
        return base, {"model": "auto_cost_rule_based", "claim_type": "auto"}
    
    else:
        # Fallback for home/travel
        base_costs = {"home": 3500.0, "travel": 800.0}
        return base_costs.get(claim_type, 1000.0), {"model": "stub"}


def acceptance_model_predict(fraud_score: float) -> tuple[float, dict]:
    """
    Predict acceptance probability based on fraud score.
    """
    # Inverse of fraud score
    acceptance = 1.0 - fraud_score
    return acceptance, {"model": "acceptance_derived_from_fraud"}


# ============ LANGGRAPH NODES ============

def run_cost_model(state: ClaimState) -> ClaimState:
    """Run the cost estimation model."""
    claim_type = state.get("claim_type", "unknown")
    structured = state.get("structured_claims", {})
    extracted_fields = structured.get(claim_type, {})
    
    # Get user profile
    user_id = "user_123"  # TODO: Get from state
    profile = get_user_profile_data(user_id, claim_type)
    
    print(f"\n[COST MODEL]")
    print(f"  Claim type: {claim_type}")
    print(f"  Extracted fields: {extracted_fields}")
    print(f"  Profile data: {profile}")
    
    cost, meta = cost_model_predict(claim_type, extracted_fields, profile)
    
    print(f"  → Predicted cost: ${cost:,.2f}")
    print(f"  → Model: {meta.get('model')}")
    
    model_meta = state.get("model_meta") or {}
    model_meta["cost"] = meta

    return {"predicted_cost": float(cost), "model_meta": model_meta}  # Convert numpy.float64 to float


def run_fraud_model(state: ClaimState) -> ClaimState:
    """Run the REAL fraud detection using fraud/agent.py."""
    claim_type = state.get("claim_type", "unknown")
    structured = state.get("structured_claims", {})
    extracted_fields = structured.get(claim_type, {})

    # Get user profile
    user_id = "user_123"  # TODO: Get from state
    profile = get_user_profile_data(user_id, claim_type)

    # Prepare claim dict for fraud agent (needs specific features)
    claim_dict = prepare_claim_for_fraud_agent(extracted_fields, profile)

    print(f"\n[FRAUD MODEL - Real Agent]")
    print(f"  Claim type: {claim_type}")
    print(f"  Fraud features: {claim_dict}")

    # Call the real fraud agent
    fraud_result = fraud_agent(claim_dict)

    print(f"  → Fraud score: {fraud_result['fraud_score']:.1%}")
    print(f"  → Risk level: {fraud_result['risk_level']}")
    print(f"  → Signals: {fraud_result['signals']}")
    print(f"  → Decision: {fraud_result['decision']}")

    model_meta = state.get("model_meta") or {}
    model_meta["fraud"] = {
        "model": "fraud_agent_v1",
        "risk_level": fraud_result["risk_level"],
        "decision": fraud_result["decision"]
    }

    return {
        "fraud_score": float(fraud_result["fraud_score"]),  # Convert numpy.float64 to float
        "fraud_risk_level": fraud_result["risk_level"],
        "fraud_signals": fraud_result["signals"],
        "fraud_decision": fraud_result["decision"],
        "model_meta": model_meta
    }


def run_acceptance_model(state: ClaimState) -> ClaimState:
    """Run the acceptance probability model based on fraud score."""
    claim_type = state.get("claim_type", "unknown")
    fraud_score = state.get("fraud_score", 0.0)

    print(f"\n[ACCEPTANCE MODEL]")
    print(f"  Claim type: {claim_type}")
    print(f"  Using fraud score: {fraud_score:.1%}")

    prob, meta = acceptance_model_predict(fraud_score)

    print(f"  → Acceptance probability: {prob:.1%}")
    print(f"  → Model: {meta.get('model')}")

    model_meta = state.get("model_meta") or {}
    model_meta["acceptance"] = meta

    return {"acceptance_probability": prob, "model_meta": model_meta}


def finalize_claim(state: ClaimState) -> ClaimState:
    """Final node - summarize the claim results."""
    claim_type = state.get("claim_type", "unknown")
    cost = state.get("predicted_cost", 0.0)
    fraud = state.get("fraud_score", 0.0)
    fraud_level = state.get("fraud_risk_level", "UNKNOWN")
    fraud_signals = state.get("fraud_signals", [])
    fraud_decision = state.get("fraud_decision", "PENDING")
    acceptance = state.get("acceptance_probability", 0.0)

    # Build signals string
    signals_str = ", ".join(fraud_signals) if fraud_signals else "None"

    reply = (
        f"\n{'='*50}\n"
        f"CLAIM PROCESSED\n"
        f"{'='*50}\n"
        f"Type: {claim_type.upper()}\n"
        f"Estimated Cost: ${cost:,.2f}\n"
        f"\n--- Fraud Analysis ---\n"
        f"Fraud Score: {fraud:.1%}\n"
        f"Risk Level: {fraud_level}\n"
        f"Signals: {signals_str}\n"
        f"Decision: {fraud_decision}\n"
        f"\nAcceptance Probability: {acceptance:.1%}\n"
        f"{'='*50}"
    )

    return {"messages": [AIMessage(content=reply)]}