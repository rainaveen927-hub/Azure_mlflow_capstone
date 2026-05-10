import json
import joblib
import pandas as pd
import os

def init():
    global model, scaler, feature_columns
    base = os.getenv("AZUREML_MODEL_DIR")
    model = joblib.load(os.path.join(base, "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(base, "feature_columns.pkl"))

def run(raw_data):
    data = json.loads(raw_data)
    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_columns, fill_value=0)

    num_cols = df.select_dtypes(include=["int64","float64"]).columns
    df[num_cols] = scaler.transform(df[num_cols])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0].tolist()

    return {"prediction": int(pred), "probability": prob}