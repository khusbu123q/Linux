import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.datasets import make_regression, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb


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


def load_regression_data():
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_classification_data():
    data = load_iris()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_xgboost_regressor():
    X_train, X_test, y_train, y_test = load_regression_data()
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"XGBoost Regressor MSE: {mse}")


def train_xgboost_classifier():
    X_train, X_test, y_train, y_test = load_classification_data()
    model = xgb.XGBClassifier(eval_metric='mlogloss', n_estimators=100)  # Removed use_label_encoder
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Classifier Accuracy: {accuracy}")


def main():
    train_xgboost_regressor()
    train_xgboost_classifier()


if __name__ == "__main__":
    main()
