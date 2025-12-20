"""
Estimation de la répartition des coûts (injury/property/vehicle)
"""
from typing import Dict

def estimate_cost_breakdown(claim: Dict, total_cost: float) -> Dict:
    """
    Estimer la répartition du coût total
    Basé sur les patterns du dataset insurance_claims.csv
    """
    # Ajuster selon les blessures corporelles
    if claim.get("bodily_injuries", 0) > 0:
        injury_ratio = 0.35 + (claim["bodily_injuries"] * 0.05)
    else:
        injury_ratio = 0.10
    
    # Ajuster selon le nombre de véhicules
    vehicle_ratio = 0.50 + (claim.get("number_of_vehicles_involved", 1) * 0.05)
    
    # Le reste va aux dommages matériels
    property_ratio = 1.0 - injury_ratio - vehicle_ratio
    property_ratio = max(0.1, property_ratio)  # Minimum 10%
    
    # Normaliser pour que la somme = 1
    total_ratio = injury_ratio + property_ratio + vehicle_ratio
    
    breakdown = {
        "injury_claim": round(total_cost * (injury_ratio / total_ratio), 2),
        "property_claim": round(total_cost * (property_ratio / total_ratio), 2),
        "vehicle_claim": round(total_cost * (vehicle_ratio / total_ratio), 2)
    }
    
    return breakdown


def identify_risk_factors(claim: Dict, predicted_cost: float) -> list:
    """
    Identifier les facteurs de risque qui influencent le coût
    """
    risk_factors = []
    
    if predicted_cost > 70000:
        risk_factors.append("HIGH_COST_CLAIM")
    
    if claim.get("bodily_injuries", 0) >= 2:
        risk_factors.append("MULTIPLE_INJURIES")
    
    if claim.get("number_of_vehicles_involved", 1) >= 3:
        risk_factors.append("MULTI_VEHICLE_COLLISION")
    
    if claim.get("witnesses", 0) == 0:
        risk_factors.append("NO_WITNESSES")
    
    if claim.get("months_as_customer", 0) < 50:
        risk_factors.append("NEW_CUSTOMER")
    
    if 0 <= claim.get("incident_hour_of_the_day", 12) <= 5:
        risk_factors.append("NIGHT_INCIDENT")
    
    return risk_factors
