# Data loading and cleaning
import pandas as pd
from sklearn.datasets import make_classification

def generate_data():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature{i}" for i in range(1, 21)])
    df['target'] = y
    return df
