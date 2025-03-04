import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def load_data():
    data = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = data["disease_score"].values
    y2 = data["disease_score_fluct"].values
    return X, y1, y2

def KFold_SciKitLearn(X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores = []  # Store R-squared values
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)  # Calculate R-squared
        r2_scores.append(r2)
    # Print R-squared values for each fold
    for i, score in enumerate(r2_scores):
        print(f"Fold {i + 1}: R^2 = {score}")
    # Print the average R-squared
    print(f"Average R^2: {sum(r2_scores) / len(r2_scores):.4f}")
    # PLOTTING R2 SCORES VS FOLDS
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(r2_scores) + 1), list(map(lambda x: round(x, 3), r2_scores)), marker="o",
             label="R^2 per fold")
    plt.xlabel("Fold")
    plt.ylabel("R^2 Score")
    plt.title("R^2 Score Across Folds")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    X,y1,y2=load_data()

    KFold_SciKitLearn(X, y1)

if __name__=="__main__":
    main()