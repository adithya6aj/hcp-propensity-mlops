# Evaluation metrics
from sklearn.metrics import classification_report
from src.data_prep import generate_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = generate_data()
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression()
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))