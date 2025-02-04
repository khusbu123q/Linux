import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def data_load():
    # Corrected variable name for the file path
    data = pd.read_csv('/home/ibab/Downloads/data.csv')

    # Extracting features and labels
    X = data.drop(["id", "diagnosis"], axis=1).values  # Assuming the data is in a proper format
    y = (data["diagnosis"] == "M").astype(int).values  # Converting 'M' -> 1, 'B' -> 0
    return X, y


def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(X, y, alpha=0.01, num_iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    bias = 0

    for _ in range(num_iterations):
        z = np.dot(X, theta) + bias
        y_pred = sigmoid(z)

        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        theta -= alpha * dw
        bias -= alpha * db

    return theta, bias


def predict(X, theta, bias):
    z = np.dot(X, theta) + bias
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)


def confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = 0, 0, 0, 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        else:
            tp += 1
    return np.array([[tn, fp], [fn, tp]])


def main():
    # Load data
    X, y = data_load()

    # Standardize the data
    X = standardize(X)

    # Split data into training and testing sets (70-30 split)
    split_ratio = 0.7
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train the logistic regression model
    theta, bias = logistic_regression(X_train, y_train, alpha=0.01, num_iterations=10000)

    # Make predictions on the test set
    y_pred = predict(X_test, theta, bias)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign', 'Malignant'])
    plt.yticks(tick_marks, ['Benign', 'Malignant'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Annotate the confusion matrix with the counts
    for i in range(2):
        for j in range(2):
            plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

    plt.show()


if __name__ == "__main__":
    main()
