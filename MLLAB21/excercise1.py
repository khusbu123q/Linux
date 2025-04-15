import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def k_means(X, n_clusters=3, max_iter=300, tol=1e-4):
    """
    K-Means clustering algorithm implementation.

    Parameters:
    - X: Input data (n_samples, n_features)
    - n_clusters: Number of clusters (default: 3)
    - max_iter: Maximum number of iterations (default: 300)
    - tol: Tolerance for convergence (default: 1e-4)

    Returns:
    - centroids: Final cluster centers
    - labels: Cluster assignments for each point
    """
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for _ in range(max_iter):
        # Calculate distances from each point to each centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))

        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=0)

        # Store previous centroids for convergence check
        old_centroids = centroids.copy()

        # Update centroids as the mean of assigned points
        for i in range(n_clusters):
            if np.any(labels == i):  # Check if cluster has points
                centroids[i] = X[labels == i].mean(axis=0)

        # Check for convergence
        if np.linalg.norm(centroids - old_centroids) < tol:
            break

    return centroids, labels


# Example usage
if __name__ == "__main__":
    # Generate sample data
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Run K-Means
    centroids, labels = k_means(X, n_clusters=4)

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title("K-Means Clustering (Procedural Implementation)")
    plt.show()