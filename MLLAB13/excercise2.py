import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


## Loading the dataset
def load_data():
    X, y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y).values.ravel()
    return X, y


# Function to create a bootstrap sample from the data (with replacement)
def bootstrap_sample(X, y, random_state=None):
    np.random.seed(random_state)
    # Generate random indices with replacement
    indices = np.random.choice(len(X), size=len(X), replace=True)
    # Use the indices to create bootstrap samples
    X_sample = X.iloc[indices]
    y_sample = y[indices]
    return X_sample, y_sample


# Bagging Regressor Implementation with DecisionTreeRegressor from scikit-learn
def bagging_regressor_fit(X_train, y_train, n_estimators=10, max_depth=5, random_state=None):
    np.random.seed(random_state)
    models = []

    for _ in range(n_estimators):
        # Create a bootstrap sample
        X_sample, y_sample = bootstrap_sample(X_train, y_train, random_state=random_state)

        # Train the base model (decision tree) on the bootstrap sample
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        model.fit(X_sample, y_sample)
        models.append(model)

    return models


# Function to predict using Bagging Regressor
def bagging_regressor_predict(X_test, models):
    predictions = np.zeros((len(X_test), len(models)))

    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_test)

    # Aggregate predictions by averaging
    return predictions.mean(axis=1)


# K-Fold Cross-Validation without scikit-learn
def k_fold_cross_validation(X, y, n_splits=5, n_estimators=50, max_depth=5, random_state=None):
    if random_state:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_splits
    mse_scores = []
    r2_scores = []

    for i in range(n_splits):
        # Define test and train indices
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Fit the Bagging Regressor with Decision Tree as base model
        models = bagging_regressor_fit(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state)

        # Make predictions on the test set
        y_pred = bagging_regressor_predict(X_test, models)

        # Calculate Mean Squared Error and R^2 score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    # Print average MSE and R^2 across all folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    print(f"Average Mean Squared Error across {n_splits} folds: {avg_mse:.4f}")
    print(f"Average R^2 score across {n_splits} folds: {avg_r2:.4f}")

    return mse_scores, r2_scores


# Main Function
def main():
    # Load data
    X, y = load_data()

    # Perform K-Fold Cross-Validation (default 5 folds)
    k_fold_cross_validation(X, y, n_splits=10, n_estimators=50, max_depth=5, random_state=42)


if __name__ == "__main__":
    main()
