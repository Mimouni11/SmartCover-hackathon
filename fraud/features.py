
import pandas as pd

FEATURES = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "capital-gains",
    "capital-loss",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim"
]


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # label
    df["fraud"] = df["fraud_reported"].map({"Y": 1, "N": 0})

    # keep numeric only
    X = df[FEATURES]
    y = df["fraud"]

    return X, y
