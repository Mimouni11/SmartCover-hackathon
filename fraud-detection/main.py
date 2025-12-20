from graph.fraud_graph import build_fraud_graph

graph = build_fraud_graph()

# Exemple de claim
sample_claim = {
    "months_as_customer": 3,
    "age": 24,
    "policy_deductable": 1000,
    "policy_annual_premium": 1200,
    "umbrella_limit": 0,
    "capital-gains": 0,
    "capital-loss": 0,
    "incident_hour_of_the_day": 2,
    "number_of_vehicles_involved": 1,
    "bodily_injuries": 1,
    "witnesses": 0,
    "total_claim_amount": 75000,
    "injury_claim": 30000,
    "property_claim": 20000,
    "vehicle_claim": 25000
}

# État initial
initial_state = {"anonymized_claim": sample_claim}

result = graph.invoke(initial_state)

# Affiche le résultat
print("\n✅ Résultat de l'analyse:")
print(result["fraud_analysis"])
