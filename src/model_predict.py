import joblib
import numpy as np

def predict_salary(model_path: str, experience_years: float):
    """Loads the trained model and predicts salary for given experience."""
    model = joblib.load(model_path)
    predicted_salary = model.predict(np.array([[experience_years]]))
    return predicted_salary[0]
