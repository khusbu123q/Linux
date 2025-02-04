import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def data_load():
    # Load the dataset
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

    # Split data into training, validation, and testing sets (60-20-20 split)
    split_ratio_train = 0.6
    split_ratio_val = 0.2
    split_index_train = int(split_ratio_train * len(X))
    split_index_val = int(split_ratio_train + split_ratio_val * len(X))

    X_train, X_val, X_test = X[:split_index_train], X[split_index_train:split_index_val], X[split_index_val:]
    y_train, y_val, y_test = y[:split_index_train], y[split_index_train:split_index_val], y[split_index_val:]

    # Hyperparameter tuning using the validation set (example: varying the learning rate)
    best_accuracy = 0
    best_alpha = 0.01

    # Try different learning rates for model selection
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        # Train the model
        theta, bias = logistic_regression(X_train, y_train, alpha=alpha, num_iterations=10000)

        # Predict on the validation set
        y_val_pred = predict(X_val, theta, bias)

        # Calculate accuracy on the validation set
        accuracy_val = np.mean(y_val_pred == y_val)

        print(f"Validation Accuracy for alpha={alpha}: {accuracy_val:.2f}")

        # Select the best model based on validation accuracy
        if accuracy_val > best_accuracy:
            best_accuracy = accuracy_val
            best_alpha = alpha
            best_theta = theta
            best_bias = bias

    # Final model evaluation with the test set using the best alpha
    print(f"Best alpha: {best_alpha}, Best validation accuracy: {best_accuracy:.2f}")

    # Test the final model on the test set
    y_test_pred = predict(X_test, best_theta, best_bias)

    # Calculate accuracy on the test set
    accuracy_test = np.mean(y_test_pred == y_test)
    print(f"Test Accuracy: {accuracy_test:.2f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

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
