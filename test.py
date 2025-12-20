import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\lenovo\OneDrive\Documents\hackaton\models\anomaly_model.pkl"

insurance_claim_obj = {
    "months_as_customer": 120,
    "age": 42,
    "policy_number": 123456,
    "policy_bind_date": "2015-06-15",
    "policy_state": "CA",
    "policy_csl": "250/500",
    "policy_deductable": 500,
    "policy_annual_premium": 1250.75,
    "umbrella_limit": 0,
    "insured_zip": 90210,
    "insured_sex": "MALE",
    "insured_education_level": "College",
    "insured_occupation": "engineer",
    "insured_hobbies": "reading",
    "insured_relationship": "husband",
    "capital-gains": 0,
    "capital-loss": 0,
    "incident_date": "2025-12-01",
    "incident_type": "Single Vehicle Collision",
    "collision_type": "Rear Collision",
    "incident_severity": "Minor Damage",
    "authorities_contacted": "Police",
    "incident_state": "CA",
    "incident_city": "Los Angeles",
    "incident_location": "Street",
    "incident_hour_of_the_day": 14,
    "number_of_vehicles_involved": 1,
    "property_damage": "YES",
    "bodily_injuries": 0,
    "witnesses": 1,
    "police_report_available": "YES",
    "total_claim_amount": 6500,
    "injury_claim": 0,
    "property_claim": 1500,
    "vehicle_claim": 5000,
    "auto_make": "Toyota",
    "auto_model": "Corolla",
    "auto_year": 2018,
    "fraud_reported": "N",
    "_c39": None,
}

model = joblib.load(MODEL_PATH)

X = pd.DataFrame([insurance_claim_obj])

# drop non-features if needed
X = X.drop(columns=["fraud_reported", "_c39"], errors="ignore")

# optional: align to training columns if the model was fit on encoded features
if hasattr(model, "feature_names_in_"):
    X = X.reindex(columns=list(model.feature_names_in_), fill_value=0)

pred = model.predict(X)
print(pred)
