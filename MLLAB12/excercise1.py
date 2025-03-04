import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import load_diabetes

def datasets():

    data = load_diabetes()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)
    # df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')

    # X = df[["age", "BMI", "BP", "blood_sugar", "Gender"]].values
    # y = df["disease_score"].values

    # Split dataset
    return train_test_split(X, y, test_size=0.2, random_state=42)


class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return np.mean(y)

        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        best_mse = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold  # Use `>` to avoid complement issues

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:  # Avoid empty partitions
                    continue

                mse = self._mse(y[left_idx]) + self._mse(y[right_idx])

                if mse < best_mse:
                    best_mse, best_feature, best_threshold = mse, feature, threshold

        return best_feature, best_threshold

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _predict(self, tree, X):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_subtree, right_subtree = tree
        if X[feature] <= threshold:
            return self._predict(left_subtree, X)
        else:
            return self._predict(right_subtree, X)

    def predict(self, X):
        return np.array([self._predict(self.tree, x) for x in X])


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = datasets()  # Fixed order

    # Training the model
    model = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Mean Squared Error
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Mean Squared Error: {mse:.2f}")




