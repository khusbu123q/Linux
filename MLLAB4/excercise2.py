import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

def load_data():
    [X, y] = fetch_california_housing(return_X_y=True)
    return X, y

def hypothesis(x, theta):
    return np.dot(x, theta)


def loss(x, y, theta):
    predictions = hypothesis(x, theta)
    error = predictions - y
    cost = (1 / (2 * len(y))) * np.sum(error**2)
    return cost


def derivative(x, y, theta):
    predictions = hypothesis(x, theta)
    error = predictions - y
    gradients = (1 / len(y)) * np.dot(x.T, error)
    return gradients


def gradient_descant(x, y, alpha=0.01, iterations=1000):
    theta = np.zeros(x.shape[1])
    cost_graph = []
    for i in range(iterations):
        costs = loss(x, y, theta)
        gradients = derivative(x, y, theta)
        theta -= alpha * gradients
        cost_graph.append(costs)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {costs:.4f}")
    return theta, cost_graph


def scale_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def r2score(y_true, y_pred):
    tot = np.sum((y_true - np.mean(y_true))**2)
    res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (res / tot)
    return r2


def train_test_split(X, y, test_size=0.30, random_state=999):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def main():
    X, y = load_data()
    X_scaled, mean, std = scale_features(X)
    X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]


    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=999)


    alpha = 0.01
    iterations = 1000


    theta, cost_graph = gradient_descant(X_train, y_train, alpha, iterations)

    print("\nTRAINING")
    print(f"N = {len(X)}")
    y_pred = hypothesis(X_test, theta)
    r2 = r2score(y_test, y_pred)
    print(f"R2 score is {r2:.2f} (close to 1 is good)")

    plt.plot(range(len(cost_graph)), cost_graph)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.title("Cost function vs. Number of iterations")
    plt.show()

if __name__ == "__main__":
    main()


