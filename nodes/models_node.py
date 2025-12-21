"""
Models Node - Cost estimation, fraud detection, and acceptance models.
Runs after extraction is complete.
"""

import pickle
import numpy as np
import json
import os
from typing import Any
from langchain_core.messages import AIMessage

from .state import ClaimState


# ============ LOAD MODELS ============

# Get the project root directory (parent of nodes/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

print(f"[DEBUG] Looking for models in: {os.path.abspath(MODELS_DIR)}")

# Load pickle models
fraud_model = None
cost_model_health = None

try:
    fraud_model_path = os.path.join(MODELS_DIR, "insurance_model.pkl")
    with open(fraud_model_path, "rb") as f:
        fraud_model = pickle.load(f)
    print(f"✓ Loaded fraud_model from {fraud_model_path}")
except FileNotFoundError as e:
    print(f"⚠️  insurance_model.pkl not found at {fraud_model_path}")
except Exception as e:
    print(f"⚠️  Error loading fraud_model: {e}")

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


def fraud_model_predict(claim_type: str, extracted_fields: dict, profile: dict) -> tuple[float, dict]:
    """
    Predict fraud probability using ML model.
    """
    if claim_type == "auto" and fraud_model is not None:
        features = prepare_auto_features(profile)
        # Assuming fraud_model returns probability
        fraud_prob = float(fraud_model.predict_proba(features)[0][1])  # probability of fraud class
        return fraud_prob, {"model": "auto_fraud_ml_v1", "features_used": len(features[0])}
    
    # Stub for other types
    return 0.08, {"model": "fraud_stub"}


def acceptance_model_predict(claim_type: str, extracted_fields: dict, profile: dict) -> tuple[float, dict]:
    """
    Predict acceptance probability.
    """
    # Stub - inverse of fraud score for now
    fraud_score, _ = fraud_model_predict(claim_type, extracted_fields, profile)
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
    
    return {"predicted_cost": cost, "model_meta": model_meta}


def run_fraud_model(state: ClaimState) -> ClaimState:
    """Run the fraud detection model."""
    claim_type = state.get("claim_type", "unknown")
    structured = state.get("structured_claims", {})
    extracted_fields = structured.get(claim_type, {})
    
    # Get user profile
    user_id = "user_123"  # TODO: Get from state
    profile = get_user_profile_data(user_id, claim_type)
    
    print(f"\n[FRAUD MODEL]")
    print(f"  Claim type: {claim_type}")
    print(f"  Extracted fields: {extracted_fields}")
    print(f"  Profile data: {profile}")
    
    fraud, meta = fraud_model_predict(claim_type, extracted_fields, profile)
    
    print(f"  → Fraud score: {fraud:.1%}")
    print(f"  → Model: {meta.get('model')}")
    
    model_meta = state.get("model_meta") or {}
    model_meta["fraud"] = meta
    
    return {"fraud_score": fraud, "model_meta": model_meta}


def run_acceptance_model(state: ClaimState) -> ClaimState:
    """Run the acceptance probability model."""
    claim_type = state.get("claim_type", "unknown")
    structured = state.get("structured_claims", {})
    extracted_fields = structured.get(claim_type, {})
    
    # Get user profile
    user_id = "user_123"  # TODO: Get from state
    profile = get_user_profile_data(user_id, claim_type)
    
    print(f"\n[ACCEPTANCE MODEL]")
    print(f"  Claim type: {claim_type}")
    
    prob, meta = acceptance_model_predict(claim_type, extracted_fields, profile)
    
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
    acceptance = state.get("acceptance_probability", 0.0)
    
    reply = (
        f"\n{'='*50}\n"
        f"CLAIM PROCESSED\n"
        f"{'='*50}\n"
        f"Type: {claim_type.upper()}\n"
        f"Estimated Cost: ${cost:,.2f}\n"
        f"Fraud Risk: {fraud:.1%}\n"
        f"Acceptance Probability: {acceptance:.1%}\n"
        f"{'='*50}"
    )
    
    return {"messages": [AIMessage(content=reply)]}