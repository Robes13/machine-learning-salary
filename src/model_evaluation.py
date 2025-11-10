import matplotlib.pyplot as plt

def plot_regression_line(X, y, model):
    """Plots the regression line along with data points."""
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, model.predict(X), color='red', label='Regression line')
    plt.title('Salary vs Experience')
    plt.xlabel('Experience (Years)')
    plt.ylabel('Salary ($)')
    plt.legend()
    plt.show()
