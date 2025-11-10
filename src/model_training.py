import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

def train_salary_model(X, y, model_path: str):
    """Trains a linear regression model and saves it to disk."""
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model
