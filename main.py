from src.data_loader import load_salary_data
from src.model_training import train_salary_model
from src.model_evaluation import plot_regression_line
from src.model_predict import predict_salary
import numpy as np

def main():
    data_path = "data/salary_data.csv"
    model_path = "models/salary_model.pkl"

    # 1️⃣ Load Data
    df = load_salary_data(data_path)
    X = df[['Experience']].values
    y = df['Salary'].values

    # 2️⃣ Train Model
    model = train_salary_model(X, y, model_path)
    print("Model trained and saved successfully!")

    # 3️⃣ Visualize Regression
    plot_regression_line(X, y, model)

    # 4️⃣ Predict a new value
    experience = 4.5
    predicted_salary = predict_salary(model_path, experience)
    print(f"Predicted salary for {experience} years of experience: ${predicted_salary:.2f}")

if __name__ == "__main__":
    main()
