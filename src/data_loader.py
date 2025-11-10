import pandas as pd

def load_salary_data(file_path: str):
    """Loads the salary dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df
