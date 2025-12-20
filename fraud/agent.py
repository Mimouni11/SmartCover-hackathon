import joblib
import numpy as np
from fraud.rules import rule_engine
from fraud.features import FEATURES


import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # hackathon/
fraud_model_path = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
anomaly_model_path = os.path.join(BASE_DIR, "models", "anomaly_model.pkl")

fraud_model = joblib.load(fraud_model_path)
anomaly_model = joblib.load(anomaly_model_path)


def fraud_agent(claim: dict):
    x = np.array([claim[f] for f in FEATURES]).reshape(1, -1)

    # ML
    ml_score = fraud_model.predict_proba(x)[0][1]

    # Anomaly
    anomaly = anomaly_model.predict(x)[0]
    anomaly_score = 0.2 if anomaly == -1 else 0

    # Rules
    rules = rule_engine(claim)
    rules_score = min(len(rules) * 0.1, 0.3)

    fraud_score = min(ml_score + anomaly_score + rules_score, 1.0)

    return {
        "fraud_score": round(fraud_score, 2),
        "risk_level": (
            "HIGH" if fraud_score > 0.7
            else "MEDIUM" if fraud_score > 0.4
            else "LOW"
        ),
        "signals": rules,
        "decision": "FLAG_FOR_REVIEW" if fraud_score > 0.5 else "AUTO_APPROVE"
    }
