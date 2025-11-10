import os
import pandas as pd
import numpy as np

#Original data
data = {
    "Experience": [1.1, 1.3, 1.5, 2.0, 2.2, 3.0, 3.2],
    "Salary": [39343, 46205, 37731, 43525, 39891, 56642, 60150]
}
df_original = pd.DataFrame(data)

#Generate 100000 rows
num_rows = 1000
experience_min, experience_max = df_original['Experience'].min(), df_original['Experience'].max()
experience_values = np.linspace(experience_min, experience_max, num_rows)

#Interpolate salary
salary_values = np.interp(experience_values, df_original['Experience'], df_original['Salary'])

#Add some random noise (+/- 10%)
noise = salary_values * 0.1 * np.random.randn(num_rows)
salary_noisy = salary_values + noise

#Create new DataFrame
df_large = pd.DataFrame({
    "Experience": np.round(experience_values, 2),
    "Salary": np.round(salary_noisy, 2)
})

#Save to CSV
df_large.to_csv("../data/salary_data.csv", index=False)

print("Generated 600-row salary dataset saved to data/salary_data_large.csv")
df_large.head(10)