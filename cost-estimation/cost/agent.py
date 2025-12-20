"""
Agent principal d'estimation de coûts
"""
import joblib
import numpy as np
from typing import Dict

# ✅ Imports absolus au lieu de relatifs
from features import extract_features, engineer_features
from breakdown import estimate_cost_breakdown, identify_risk_factors

# Charger les modèles au démarrage
cost_model = joblib.load("../models/cost_model.pkl")
cost_scaler = joblib.load("../models/cost_scaler.pkl")


def cost_estimation_agent(claim: Dict) -> Dict:
    """
    Agent complet d'estimation de coûts
    
    Args:
        claim: Dictionnaire avec les données de la réclamation
        
    Returns:
        Dictionnaire avec l'estimation complète
    """
    # 1. Feature engineering
    base_features = extract_features(claim)
    feature_vector = engineer_features(base_features)
    
    # 2. Normalisation
    normalized_features = cost_scaler.transform(feature_vector.reshape(1, -1))
    
    # 3. Prédiction XGBoost
    predicted_cost = float(cost_model.predict(normalized_features)[0])
    
    # 4. Intervalle de confiance
    uncertainty = predicted_cost * 0.15  # ±15%
    confidence_interval = {
        "min": max(0, predicted_cost - uncertainty),
        "max": predicted_cost + uncertainty,
        "confidence_level": 0.85
    }
    
    # 5. Répartition des coûts
    cost_breakdown = estimate_cost_breakdown(claim, predicted_cost)
    
    # 6. Analyse des facteurs de risque
    risk_factors = identify_risk_factors(claim, predicted_cost)
    
    return {
        "predicted_cost": round(predicted_cost, 2),
        "confidence_interval": confidence_interval,
        "cost_breakdown": cost_breakdown,
        "risk_factors": risk_factors
    }


# ✅ Pour tester directement le fichier
if __name__ == "__main__":
    # Exemple de test
    test_claim = {
        "months_as_customer": 328,
        "age": 48,
        "policy_deductable": 1000,
        "policy_annual_premium": 1406.91,
        "umbrella_limit": 0,
        "capital-gains": 53300,
        "capital-loss": 0,
        "incident_hour_of_the_day": 5,
        "number_of_vehicles_involved": 1,
        "bodily_injuries": 1,
        "witnesses": 2
    }
    
    result = cost_estimation_agent(test_claim)
    print("\n✅ Résultat:")
    print(f"Coût estimé: ${result['predicted_cost']:,.2f}")
    print(f"Facteurs de risque: {result['risk_factors']}")
