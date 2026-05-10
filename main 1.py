from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("outputs/random_forest_model.pkl")
scaler = joblib.load("outputs/scaler.pkl")
feature_columns = joblib.load("outputs/feature_columns.pkl")

app = FastAPI(title="Loan Default Prediction API")

class LoanInput(BaseModel):
    data: dict

@app.get("/")
def home():
    return {"status": "API Running"}

@app.post("/predict")
def predict(request: LoanInput):
    df = pd.DataFrame([request.data])
    df = df.reindex(columns=feature_columns, fill_value=0)

    num_cols = df.select_dtypes(include=["int64","float64"]).columns
    df[num_cols] = scaler.transform(df[num_cols])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0].tolist()

    return {"prediction": int(pred), "probability": prob}