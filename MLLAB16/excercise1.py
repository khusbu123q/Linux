import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class BaggingRegressor:
    def __init__(self, n_estimators=10, max_samples=0.8):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []

    def fit(self, X, y):
        self.models = []
        n_samples = int(self.max_samples * len(X))
        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(X, y, n_samples=n_samples, random_state=None)
            model = DecisionTreeRegressor()
            model.fit(X_resampled, y_resampled)
            self.models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)


def load_data():
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def main():
    X_train, X_test, y_train, y_test = load_data()

    bagging_regressor = BaggingRegressor(n_estimators=10, max_samples=0.8)
    bagging_regressor.fit(X_train, y_train)

    y_pred = bagging_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")


if __name__ == "__main__":
    main()
