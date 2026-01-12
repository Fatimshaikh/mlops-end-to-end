from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="ML Model API")

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Request schema
class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "ML Model API is running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
