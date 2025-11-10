import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

#Load data
data_path = "data/salary_data.csv"
df = pd.read_csv(data_path)
X = df[['Experience']].values
y = df['Salary'].values

train_mse_list = []
test_mse_list = []
random_states = range(0, 50)

best_test_mse = float("inf")
best_random_state = None
best_model = None

for rs in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    mse_train = mean_squared_error(y_train, model.predict(X_train))
    mse_test = mean_squared_error(y_test, model.predict(X_test))
    
    train_mse_list.append(mse_train)
    test_mse_list.append(mse_test)
    
    #Track the model with the lowest test MSE
    if mse_test < best_test_mse:
        best_test_mse = mse_test
        best_random_state = rs
        best_model = model

#Save the best model
joblib.dump(best_model, "models/test_salary_model.pkl")
print(f"Saved model with best test MSE: {best_test_mse:.2f} (random_state={best_random_state})")

#Plot train vs test MSE
plt.figure(figsize=(10,6))
plt.plot(random_states, train_mse_list, label='Train MSE', marker='o')
plt.plot(random_states, test_mse_list, label='Test MSE', marker='o')
plt.axvline(best_random_state, color='green', linestyle='--', label='Best Random State')
plt.xlabel("Random State")
plt.ylabel("MSE")
plt.title("Train vs Test MSE for different Random States")
plt.legend()
plt.grid(True)
plt.show()
