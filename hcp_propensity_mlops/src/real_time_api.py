# Real-time scoring via FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import yaml
import numpy as np

class InputFeatures(BaseModel):
    features: list

app = FastAPI()

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config['tracking_uri'])
model = mlflow.sklearn.load_model(f"models:/{config['model_name']}/latest")

@app.post("/predict")
def predict(input: InputFeatures):
    x = np.array(input.features).reshape(1, -1)
    pred = model.predict_proba(x)[0][1]
    return {"score": pred}