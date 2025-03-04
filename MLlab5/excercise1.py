import numpy as np
import pandas as pd

def hypothesis(x, theta):
    return np.dot(x, theta)

def stochastic_gradient_descent(X, y, alpha=0.01, iteration=100):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    bias = 0

    for iterations in range(iteration):
        for i in range(n_samples):
            y_pred = np.dot(X[i], theta) + bias
            error = y_pred - y[i]


            data_weights = error * X[i]
            data_bias = error


            theta -= alpha * data_weights
            bias -= alpha * data_bias


        cost = np.mean((np.dot(X, theta) + bias - y) ** 2)
        print(f"Iteration {iterations + 1}/{iteration}, Cost: {cost:.4f}")

    return theta,bias

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

def scale_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

if __name__ == "__main__":

    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    X = df[["age", "BMI", "BP", "blood_sugar", "Gender"]].values
    y = df["disease_score"].values


    X_scaled, mean, std = scale_features(X)
    X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]


    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=999)


    alpha = 0.001
    iteration = 1000
    weights, bias = stochastic_gradient_descent(X_train, y_train, alpha, iteration)

    print("Trained Weights:", weights)
    print("Trained Bias:", bias)


    y_pred = hypothesis(X_test, weights) + bias
    r2 = r2score(y_test, y_pred)
    print(f"R2 score is {r2:.2f} (close to 1 is good)")

