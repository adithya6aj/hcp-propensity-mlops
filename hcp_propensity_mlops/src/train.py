import mlflow
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hcp_propensity_mlops.src.data_prep import generate_data
with open("hcp_propensity_mlops/config/config.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config['tracking_uri'])
mlflow.set_experiment(config['experiment_name'])

df = generate_data()
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(accuracy)

with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, config['model_name'])
