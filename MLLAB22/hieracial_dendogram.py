
##USArrests dataset hierarchial for Eucledian distance and correlation based distance (cutting at 3)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from statsmodels.datasets import get_rdataset


def load_and_scale_data():
    # df = pd.read_csv('C:/Users/Aritri Baidya/Downloads/USArrests.csv', index_col=0)
    df = get_rdataset('USArrests').data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return df, scaled_data


def plot_dendrogram(linked, labels, title, ylabel):
    plt.figure(figsize=(12, 6))
    dendrogram(linked, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.xlabel("States")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def cluster_and_print(df, linkage_matrix, num_clusters, label_col):
    clusters = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')
    df[label_col] = clusters
    for i in range(1, num_clusters + 1):
        print(f"\n{label_col} {i} States:")
        print(df[df[label_col] == i].index.tolist())


def compare_correlation_euclidean():
    # Load data
    # df = pd.read_csv('C:/Users/Aritri Baidya/Downloads/USArrests.csv', index_col=0)
    df = get_rdataset('USArrests').data
    df_numeric = df.select_dtypes(include=[np.number])
    X = StandardScaler().fit_transform(df_numeric)

    # Euclidean distances (squared)
    euclidean_dist = pairwise_distances(X, metric='euclidean') ** 2

    # Correlation distance: 1 - correlation
    corr_matrix = np.corrcoef(X)
    correlation_dist = 1 - corr_matrix

    # Flatten upper triangle
    euclidean_flat = euclidean_dist[np.triu_indices_from(euclidean_dist, k=1)]
    correlation_flat = correlation_dist[np.triu_indices_from(correlation_dist, k=1)]

    # Plot
    plt.figure(figsize=(8, 6))
    sns.regplot(x=correlation_flat, y=euclidean_flat, scatter_kws={'s': 20})
    plt.xlabel("1 - Correlation (Correlation Distance)")
    plt.ylabel("Squared Euclidean Distance")
    plt.title("Correlation Distance vs. Squared Euclidean Distance")
    plt.grid(True)
    plt.show()

    # Correlation between distance matrices
    corr_between = np.corrcoef(correlation_flat, euclidean_flat)[0, 1]
    print(f"(e) Pearson correlation between distance matrices: {corr_between:.4f}")


def main():
    # Load and scale data
    df, scaled_data = load_and_scale_data()
    labels = df.index.tolist()

    # --- (a) Hierarchical clustering with Euclidean distance ---
    linked_euclidean = linkage(scaled_data, method='complete', metric='euclidean')
    print("Plotting Euclidean Distance Dendrogram...")
    plot_dendrogram(linked_euclidean, labels, "Complete Linkage (Euclidean Distance)", "Euclidean Distance")

    # --- (b) Cut dendrogram for Euclidean distance ---
    print("\n=== Euclidean Distance Clustering ===")
    cluster_and_print(df, linked_euclidean, 3, 'EuclideanCluster')

    # --- (c) Hierarchical clustering with Correlation distance ---
    corr_distance = pdist(scaled_data, metric='correlation')
    linked_corr = linkage(corr_distance, method='complete')
    print("\nPlotting Correlation Distance Dendrogram...")
    plot_dendrogram(linked_corr, labels, "Complete Linkage (Correlation Distance)", "1 - Correlation")

    # --- (d) Cut dendrogram for Correlation distance ---
    print("\n=== Correlation Distance Clustering ===")
    cluster_and_print(df, linked_corr, 3, 'CorrelationCluster')
    compare_correlation_euclidean()


if __name__ == "__main__":
    main()
