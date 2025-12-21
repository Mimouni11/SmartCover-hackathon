import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from features import preprocess

# Charger les données
df = pd.read_csv("../data/insurance_claims.csv")

# Préprocessing
X, y = preprocess(df)

# Modèle fraude supervisé
fraud_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
fraud_model.fit(X, y)

# Modèle anomalies
anomaly_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)
anomaly_model.fit(X)

# Sauvegarde
joblib.dump(fraud_model, "../models/fraud_model.pkl")
joblib.dump(anomaly_model, "../models/anomaly_model.pkl")

print("Models trained and saved")
