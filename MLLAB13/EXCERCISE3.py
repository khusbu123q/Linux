from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


# Load Iris dataset
def load_data_iris():
    data = load_iris()
    X = pd.DataFrame(data.data)
    y = pd.Series(data.target)
    return X, y


# Load Diabetes dataset
def load_data_diabetes():
    data = load_diabetes()
    X = pd.DataFrame(data.data)
    y = pd.Series(data.target)
    return X, y


# Random Forest Regressor for Diabetes Dataset
def scikit_randomforest_regression():
    X, y = load_data_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3434, test_size=0.30)

    # Standardizing the data
    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)

    # Creating and fitting the model
    model = RandomForestRegressor(n_estimators=10, max_depth=2, min_samples_split=5, random_state=22)
    model.fit(X_train_scaled, y_train)

    # Making predictions
    y_pred = model.predict(X_test_scaled)
    print("Random Forest - Diabetes")
    print(f"R^2 value: {r2_score(y_test, y_pred):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=10, scoring="r2")
    print(f"R^2 mean (CV): {np.mean(cv_scores):.4f}")
    print(f"R^2 standard deviation (CV): {np.std(cv_scores):.4f}")


# Random Forest Classifier for Iris Dataset
def scikit_randomforest_classification():
    X, y = load_data_iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3434, test_size=0.30)

    # Standardizing the data
    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)

    # Creating and fitting the model
    model = RandomForestClassifier(n_estimators=10, max_depth=2, min_samples_split=5, random_state=22)
    model.fit(X_train_scaled, y_train)

    # Making predictions
    y_pred = model.predict(X_test_scaled)
    print("Random Forest - Iris")
    print(f"Accuracy value: {accuracy_score(y_test, y_pred):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    print(f"Accuracy mean (CV): {np.mean(cv_scores):.4f}")
    print(f"Accuracy standard deviation (CV): {np.std(cv_scores):.4f}")


# Run both models
scikit_randomforest_regression()
scikit_randomforest_classification()
