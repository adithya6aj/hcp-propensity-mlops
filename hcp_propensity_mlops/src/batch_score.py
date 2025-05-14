# Batch inference logic
import mlflow
import yaml
from src.data_prep import generate_data

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config['tracking_uri'])
model = mlflow.sklearn.load_model(f"models:/{config['model_name']}/latest")
df = generate_data().drop("target", axis=1)
preds = model.predict_proba(df)[:, 1]
print(preds)