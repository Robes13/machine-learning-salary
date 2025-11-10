from src.data_loader import load_salary_data
from src.model_training import train_salary_model
from src.model_evaluation import plot_regression_line
from src.model_predict import predict_salary
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main():
    data_path = "data/salary_data.csv"
    default_model_path = "models/salary_model.pkl"
    best_model_path = "models/test_salary_model.pkl"

    #Load data
    df = load_salary_data(data_path)
    X = df[['Experience']].values
    y = df['Salary'].values

    #Split for evaluation metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Load best model if available
    if os.path.exists(best_model_path):
        model_path = best_model_path
        model = joblib.load(model_path)
        print(f"Loaded best test-MSE model from {model_path}")
    else:
        model_path = default_model_path
        model = train_salary_model(X, y, model_path)
        print(f"Trained new model and saved to {model_path}")

    #Visualize regression
    plot_regression_line(X, y, model)

    #Evaluate model metrics
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print("\nModel Evaluation Metrics:")
    print(f"Train MSE: {mse_train:.2f}")
    print(f"Test MSE: {mse_test:.2f}")
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²: {r2_test:.4f}")

    #Predict salary for a specific experience
    experience = 4.5
    predicted_salary = predict_salary(model_path, experience)
    print(f"\nPredicted salary for {experience} years of experience: ${predicted_salary:.2f}")

if __name__ == "__main__":
    main()
