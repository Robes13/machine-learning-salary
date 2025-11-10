import matplotlib.pyplot as plt

def plot_data(df):
    """Plots raw experience vs salary data."""
    plt.scatter(df['Experience'], df['Salary'], color='blue')
    plt.title('Salary Data')
    plt.xlabel('Experience (Years)')
    plt.ylabel('Salary ($)')
    plt.show()
