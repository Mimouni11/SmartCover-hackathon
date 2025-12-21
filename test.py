import pickle
import pandas as pd
import numpy as np

one_house = {
    "house_age": 12,
    "house_size": 140.0,
    "num_rooms": 4,
    "num_bathrooms": 2,
    "location": "rural",
    "building_material": "wood",
    "security_features": "alarm",
    "previous_claims": 1,
    'claims_age_ratio':12,
    'age_category':'new', 
    'has_claims':1, 
    'rooms_per_bathroom':2
}

model = pickle.load(open("C:\\Users\\lenovo\\OneDrive\\Documents\\hackaton\\models\\insurance_model.pkl", "rb"))

X = pd.DataFrame([one_house])


pred = model.predict(X)
print(float(pred[0]))
