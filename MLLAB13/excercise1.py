
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


# Data preprocessing for classification
def Data_processing_class(X_Train, X_Test, y_Train, y_Test):
    label = LabelEncoder()
    y_encoded_train = label.fit_transform(y_Train)
    y_encoded_test = label.transform(y_Test)
    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_Train)
    X_test_scaled = scaling.transform(X_Test)
    return X_train_scaled, X_test_scaled, y_encoded_train, y_encoded_test


# Data preprocessing for regression
def Data_processing_reg(X_Train, X_Test, y_Train, y_Test):
    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_Train)
    X_test_scaled = scaling.transform(X_Test)
    return X_train_scaled, X_test_scaled, np.array(y_Train), np.array(y_Test)


# Bagging Regressor
def scikit_regressor():
    # Load diabetes dataset
    data = load_diabetes()
    X, y = pd.DataFrame(data.data), pd.Series(data.target)

    # Split the data into training and testing sets
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.30, random_state=22)
    X_Train_t, X_Test_t, y_Train_t, y_Test_t = Data_processing_reg(X_Train, X_Test, y_Train, y_Test)

    # Create the base regressor
    regrModel = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10, random_state=0)
    regrModel.fit(X_Train_t, y_Train_t)

    # Make predictions on the test set
    y_pred = regrModel.predict(X_Test_t)
    # Calculate R^2 score
    r2 = r2_score(y_Test_t, y_pred)
    print("Whole tree R^2 value:")
    print(f"R^2 value: {r2:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(regrModel, X, y, cv=10, scoring="r2")
    print(f"R^2 score mean (CV): {np.mean(cv_scores):.4f}")
    print(f"R^2 standard deviation (CV): {np.std(cv_scores):.4f}")


# Bagging Classifier
def scikit_classifier():
    # Load iris dataset
    data = load_iris()
    X, y = pd.DataFrame(data.data), pd.Series(data.target)

    # Split the data into training and testing sets
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.30, random_state=22)

    # Create the base classifier
    classModel = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=5, random_state=232),
        n_estimators=10,
        random_state=0,
    )
    classModel.fit(X_Train, y_Train)

    # Make predictions on the test set
    y_pred = classModel.predict(X_Test)
    # Calculate accuracy
    acc = accuracy_score(y_Test, y_pred)
    print("Whole tree accuracy value:")
    print(f"Accuracy: {acc:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(classModel, X, y, cv=10, scoring="accuracy")
    print(f"Accuracy mean (CV): {np.mean(cv_scores):.4f}")
    print(f"Accuracy standard deviation (CV): {np.std(cv_scores):.4f}")


# Main function
def main():
    scikit_regressor()
    scikit_classifier()


if __name__ == "__main__":
    main()
