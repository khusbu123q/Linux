import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Transform(x1, x2):
    """Transforms 2D points (x1, x2) into 3D feature space."""
    return np.array([x1 ** 2, np.sqrt(2) * x1 * x2, x2 ** 2]).T


def load_data():
    """Loads the dataset."""
    return np.array([
        [1, 13, 'Blue'], [1, 18, 'Blue'], [2, 9, 'Blue'], [3, 6, 'Blue'], [6, 3, 'Blue'],
        [9, 2, 'Blue'], [13, 1, 'Blue'], [18, 1, 'Blue'], [3, 15, 'Red'], [6, 6, 'Red'],
        [6, 11, 'Red'], [9, 5, 'Red'], [10, 10, 'Red'], [11, 5, 'Red'], [12, 6, 'Red'],
        [16, 3, 'Red']
    ])


def plot_2d(samples):
    """Plots the original 2D data points."""
    plt.figure(figsize=(8, 6))
    for x1, x2, label in samples:
        color = 'blue' if label == 'Blue' else 'red'
        plt.scatter(float(x1), float(x2), color=color)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Original 2D Data')
    plt.show()


def plot_3d(transformed_data, labels):
    """Plots the transformed 3D data points."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(transformed_data)):
        color = 'blue' if labels[i] == 'Blue' else 'red'
        ax.scatter(transformed_data[i, 0], transformed_data[i, 1], transformed_data[i, 2], color=color)
    ax.set_xlabel('X1^2')
    ax.set_ylabel('sqrt(2)*X1*X2')
    ax.set_zlabel('X2^2')
    plt.title('Transformed 3D Data')
    plt.show()


def polynomial_kernel(a, b):
    """Polynomial kernel function."""
    return a[0] ** 2 * b[0] ** 2 + 2 * a[0] * b[0] * a[1] * b[1] + a[1] ** 2 * b[1] ** 2


def main():
    samples = load_data()
    X1 = samples[:, 0].astype(float)
    X2 = samples[:, 1].astype(float)
    labels = samples[:, 2]

    plot_2d(samples)

    transformed_data = Transform(X1, X2)
    plot_3d(transformed_data, labels)

    x1 = np.array([3, 6])
    x2 = np.array([10, 10])
    x1_trans = Transform(x1[0], x1[1])
    x2_trans = Transform(x2[0], x2[1])
    dot_product = np.dot(x1_trans, x2_trans)
    print(f"Dot product in transformed space: {dot_product}")

    kernel_value = polynomial_kernel(x1, x2)
    print(f"Polynomial kernel value: {kernel_value}")


if __name__ == "__main__":
    main()
