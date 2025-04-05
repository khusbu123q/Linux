import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import seaborn as sns


def rbf_kernel(a, b, gamma=0.1):
    """Radial Basis Function (RBF) kernel implementation."""
    return np.exp(-gamma * np.linalg.norm(a - b) ** 2)


def polynomial_kernel(a, b):
    """Polynomial kernel function."""
    return a[0] ** 2 * b[0] ** 2 + 2 * a[0] * b[0] * a[1] * b[1] + a[1] ** 2 * b[1] ** 2


def load_custom_data():
    """Loads the custom dataset."""
    return np.array([
        [6, 5, 'Blue'], [6, 9, 'Blue'], [8, 6, 'Red'], [8, 8, 'Red'], [8, 10, 'Red'],
        [9, 2, 'Blue'], [9, 5, 'Red'], [10, 10, 'Red'], [10, 13, 'Blue'], [11, 5, 'Red'],
        [11, 8, 'Red'], [12, 6, 'Red'], [12, 11, 'Blue'], [13, 4, 'Blue'], [14, 8, 'Blue']
    ])


def train_svm_iris():
    """Train an SVM on the iris dataset with class 1 and 2 using the first two features."""
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target
    X = X[y < 2]  # Select classes 0 and 1
    y = y[y < 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM classification accuracy on Iris dataset: {accuracy:.2f}")

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=['blue', 'red'])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Boundary for Iris Classes 0 and 1")
    plt.show()


def train_svm_twitter():
    """Train an SVM for Twitter sentiment analysis using different kernels."""
    # Sample Twitter sentiment data
    tweets = [
        "I love this movie!", "This is a great product!", "Amazing experience!",
        "I hate this place.", "Worst service ever!", "Not a good experience.",
        "Absolutely fantastic!", "So disappointed with this.", "Will never buy again!"
    ]
    labels = [1, 1, 1, 0, 0, 0, 1, 0, 0]  # 1 for positive, 0 for negative

    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2, random_state=42)

    # Train SVM with different kernels
    for kernel in ['linear', 'poly', 'rbf']:
        model = make_pipeline(TfidfVectorizer(), SVC(kernel=kernel))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM with {kernel} kernel accuracy on Twitter sentiment: {accuracy:.2f}")


def main():
    train_svm_iris()
    train_svm_twitter()


if __name__ == "__main__":
    main()
