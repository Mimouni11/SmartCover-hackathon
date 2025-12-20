"""
Feature engineering pour l'estimation de coûts
"""
import numpy as np
from typing import Dict, List

def extract_features(claim: Dict) -> Dict:
    """
    Extraire les features de base de la réclamation
    """
    features = {
        'months_as_customer': claim.get('months_as_customer', 0),
        'age': claim.get('age', 0),
        'policy_deductable': claim.get('policy_deductable', 0),
        'policy_annual_premium': claim.get('policy_annual_premium', 0),
        'umbrella_limit': claim.get('umbrella_limit', 0),
        'capital_gains': claim.get('capital-gains', 0),
        'capital_loss': claim.get('capital-loss', 0),
        'incident_hour': claim.get('incident_hour_of_the_day', 0),
        'vehicles_involved': claim.get('number_of_vehicles_involved', 1),
        'bodily_injuries': claim.get('bodily_injuries', 0),
        'witnesses': claim.get('witnesses', 0),
    }
    return features


def engineer_features(base_features: Dict) -> np.ndarray:
    """
    Créer des features dérivées
    """
    # Features dérivées
    derived = {}
    derived['customer_loyalty_score'] = min(base_features['months_as_customer'] / 100, 1.0)
    derived['injury_severity'] = base_features['bodily_injuries'] * 1.5
    derived['complexity_score'] = (
        base_features['vehicles_involved'] * 0.3 + 
        base_features['bodily_injuries'] * 0.5 + 
        (1 if base_features['witnesses'] == 0 else 0) * 0.2
    )
    derived['policy_coverage_ratio'] = (
        base_features['umbrella_limit'] / max(base_features['policy_annual_premium'], 1)
    )
    derived['night_incident'] = 1 if 0 <= base_features['incident_hour'] <= 6 else 0
    
    # Ordre des features pour le modèle
    feature_order = [
        'months_as_customer', 'age', 'policy_deductable', 
        'policy_annual_premium', 'umbrella_limit', 'capital_gains',
        'capital_loss', 'incident_hour', 'vehicles_involved',
        'bodily_injuries', 'witnesses', 'customer_loyalty_score',
        'injury_severity', 'complexity_score', 'policy_coverage_ratio',
        'night_incident'
    ]
    
    # Combiner features de base et dérivées
    all_features = {**base_features, **derived}
    feature_vector = np.array([all_features[k] for k in feature_order])
    
    return feature_vector


def get_feature_names() -> List[str]:
    """
    Retourner les noms des features dans l'ordre
    """
    return [
        'months_as_customer', 'age', 'policy_deductable', 
        'policy_annual_premium', 'umbrella_limit', 'capital_gains',
        'capital_loss', 'incident_hour', 'vehicles_involved',
        'bodily_injuries', 'witnesses', 'customer_loyalty_score',
        'injury_severity', 'complexity_score', 'policy_coverage_ratio',
        'night_incident'
    ]
