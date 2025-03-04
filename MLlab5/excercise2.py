import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def load_data():
    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    X = df[["age", "BMI", "BP", "blood_sugar", "Gender"]].values
    y = df["disease_score"].values
    return X, y


X_data, y_data = load_data()


x_values = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(x_values)
sigmoid_derivative_values = sigmoid_derivative(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, sigmoid_values, label="Sigmoid Function", color="blue")
plt.plot(x_values, sigmoid_derivative_values, label="Sigmoid Derivative", color="green", linestyle="--")
plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="y=0")
plt.title("Sigmoid Function and its Derivative")
plt.xlabel("x")
plt.ylabel("Values")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
