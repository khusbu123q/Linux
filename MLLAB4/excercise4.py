import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def normal_equation(X, y):
    X = np.c_[np.ones((X.shape[0])), X]
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


def hypothesis(X, theta):
    return np.dot(X, theta)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error**2)
    return cost


def compute_gradient(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    error = predictions - y
    gradients = (1 / m) * np.dot(X.T, error)
    return gradients


def gradient_descent(X, y, alpha=0.0001,iteration=15000):
    theta = np.zeros(X.shape[1])
    cost_history = []
    for i in range(iteration):
        cost = compute_cost(X, y, theta)
        gradients = compute_gradient(X, y, theta)
        theta -= alpha * gradients
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    return theta, cost_history


def train_test_split(X, y, test_size=0.30, random_state=999):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def r2_score(y_true, y_pred):
    total_variance = np.sum((y_true - np.mean(y_true))**2)
    residual_variance = np.sum((y_true - y_pred)**2)
    r2 = 1 - (residual_variance / total_variance)
    return r2


def plot_results(X, y,y_pred_gd,y_pred_ne,y_pred_sklearn,feature_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X, y_pred_gd, color="orange", label="Gradient Descent Line", linewidth=2)
    plt.plot(X, y_pred_ne, color='red', label='Normal Equation Line', linewidth=2)
    plt.plot(X, y_pred_sklearn, color='green', label='Scikit-learn Line', linewidth=2, linestyle='dashed')
    plt.xlabel(feature_name)
    plt.ylabel('Target (Scaled Disease Score)')
    plt.title(f'Regression Line Comparison: {feature_name}')
    plt.legend()
    plt.show()

def main():

    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    X = df["age"].values.reshape(-1, 1)
    y = df["disease_score"].values * 100

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)


    X_train_aug= np.c_[np.ones((X_train.shape[0])), X_train]
    X_test_aug = np.c_[np.ones((X_test.shape[0])), X_test]
    X_aug = np.c_[np.ones((X.shape[0])), X]

    # Gradient Descent
    alpha = 0.0001
    iterations =120000
    theta_gd, cost_history = gradient_descent(X_train_aug, y_train, alpha, iterations)
    y_pred_gd = hypothesis(X_aug, theta_gd)

    # Normal Equation
    theta_ne = normal_equation(X_train, y_train)
    y_pred_ne = hypothesis(X_aug, theta_ne)

    # Scikit-learn Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_sklearn = model.predict(X)

    # Model Evaluation
    print("\nTRAINING")
    print(f"N = {len(X)}")
    r2_ne = r2_score(y_test, hypothesis(X_test_aug, theta_ne))
    r2_sklearn = r2_score(y_test, model.predict(X_test))
    r2_gd = r2_score(y_test, hypothesis(X_test_aug, theta_gd))
    print(f"Normal Equation R2 score: {r2_ne:.2f}")
    print(f"Scikit-learn R2 score: {r2_sklearn:.2f}")
    print(f"Gradient Descent R2 score: {r2_gd:.2f}")


    plot_results(X, y,y_pred_gd,y_pred_ne,y_pred_sklearn,"age")

if __name__ == "__main__":
    main()
