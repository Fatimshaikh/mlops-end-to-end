import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model & scaler
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("data/processed/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="Customer Churn Prediction API")

# Input schema
class ChurnInput(BaseModel):
    age: int
    monthly_charges: int
    tenure: int
    total_charges: int

@app.post("/predict")
def predict_churn(data: ChurnInput):
    input_df = pd.DataFrame([data.dict()])
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)

    return {
        "churn_prediction": int(prediction[0])
    }
