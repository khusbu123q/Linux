# Import required libraries
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert multi-class to binary classification (only class 0 and 1 for simplicity)
X, y = X[y != 2], y[y != 2]  # Use only class 0 and 1
y = np.where(y == 0, -1, 1)  # Convert labels to {-1, 1}


# Define a simple Decision Stump (one-level decision tree)
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        # Iterate over all features and thresholds to find the best split
        for feature in range(n_features):
            feature_values = np.sort(np.unique(X[:, feature]))
            for threshold in feature_values:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X[:, feature] < threshold] = -1
                    else:
                        predictions[X[:, feature] > threshold] = -1

                    # Compute weighted error
                    errors = sample_weights * (predictions != y)
                    error = np.sum(errors)

                    # Select the best split
                    if error < min_error:
                        min_error = error
                        self.feature_index = feature
                        self.threshold = threshold
                        self.polarity = polarity

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] > self.threshold] = -1
        return predictions


# Define AdaBoost Classifier from scratch
class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Create and train a decision stump
            stump = DecisionStump()
            stump.fit(X, y, sample_weights)
            predictions = stump.predict(X)

            # Calculate error and alpha
            error = np.sum(sample_weights * (predictions != y)) / np.sum(sample_weights)
            alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))  # Avoid division by zero

            # Update sample weights
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            # Save model and alpha
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        n_samples = X.shape[0]
        final_predictions = np.zeros(n_samples)

        # Make predictions using all weak learners
        for alpha, model in zip(self.alphas, self.models):
            predictions = model.predict(X)
            final_predictions += alpha * predictions

        # Return final predictions
        return np.sign(final_predictions)


# K-Fold Cross-Validation without scikit-learn
def k_fold_cross_validation(X, y, n_splits=5, random_state=None):
    if random_state:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_splits
    accuracies = []

    for i in range(n_splits):
        # Define test and train indices
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Instantiate and train AdaBoost classifier
        adaboost_clf = AdaBoost(n_estimators=50)
        adaboost_clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = adaboost_clf.predict(X_test)

        # Calculate accuracy
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        accuracies.append(accuracy)

    # Print average accuracy across all folds
    avg_accuracy = np.mean(accuracies)
    print(f"Average Accuracy across {n_splits} folds: {avg_accuracy * 100:.2f}%")

    return accuracies


# Perform 5-Fold Cross-Validation
k_fold_cross_validation(X, y, n_splits=5, random_state=42)
