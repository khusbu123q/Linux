import pandas as pd
import numpy as np

def load_data():
    data = pd.read_csv('/home/ibab/Downloads/data.csv')

    # Extract the "texture_mean" column and reshape it into a 2D array
    X = data["texture_mean"].values.reshape(-1, 1)
    return X

def standardize(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    return (X - X_mean) / X_std

def min_max_scaling(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

def compute_variance(X):
    return np.var(X, axis=0)


def main():
    X= load_data()

    # Normalize and standardize the data
    X_normalized = min_max_scaling(X)
    print("Min values before normalization :\n",np.min(X,axis=0))
    print("Min values after normalization:\n", np.min(X_normalized, axis=0))
    print("Max values before normalization:\n",np.max(X,axis=0))
    print("Max values after normalization:\n", np.max(X_normalized, axis=0))
    X_standardized = standardize(X)
    print("Mean values before standarization:\n", np.mean(X, axis=0))
    print("Mean values after standarization:\n", np.mean(X_standardized, axis=0))

    variance_normalized = compute_variance(X_normalized)
    variance_standardized = compute_variance(X_standardized)

    print("Variance after normalization:\n", variance_normalized)
    print("Variance after standardization:\n", variance_standardized)

if __name__=="__main__":
    main()