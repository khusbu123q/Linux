import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from time import time


def load_and_preprocess_data():
    """Load and preprocess the NCI60 gene expression data."""
    try:
        nci_data = fetch_openml(name='nci60', version=1, as_frame=True)
        X = nci_data.data
        y = nci_data.target
    except:
        # Fallback to synthetic data if real data not available
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=64, n_features=6830,
                                   n_informative=50, n_classes=8,
                                   random_state=42)
        print("Using synthetic data as NCI60 dataset not available")

    # Preprocess the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42)

    print(f"\nOriginal data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    return X_train, X_test, y_train, y_test, le


def hierarchical_feature_selection(X, y, n_clusters=50):
    """
    Select features using hierarchical clustering.

    1. Perform hierarchical clustering on features
    2. Select one representative feature from each cluster
    3. Use ANOVA F-value to select the best feature in each cluster
    """
    # Transpose because we want to cluster features (columns)
    corr_matrix = np.corrcoef(X.T)

    # Perform hierarchical clustering
    Z = linkage(corr_matrix, method='average', metric='correlation')

    # Form flat clusters
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

    # Select best feature from each cluster
    selected_features = []
    for cluster_id in np.unique(clusters):
        features_in_cluster = np.where(clusters == cluster_id)[0]
        if len(features_in_cluster) > 0:
            # Use ANOVA F-value to select best feature in cluster
            selector = SelectKBest(f_classif, k=1)
            selector.fit(X[:, features_in_cluster], y)
            best_feature = features_in_cluster[selector.get_support(indices=True)[0]]
            selected_features.append(best_feature)

    return np.array(selected_features)


def apply_pca(X_train, X_test, n_components=50):
    """Apply PCA dimensionality reduction."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.show()

    return X_train_pca, X_test_pca


def train_and_evaluate(X_train, X_test, y_train, y_test, method_name, le):
    """Train a classifier and evaluate performance."""
    start_time = time()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{method_name} Results:")
    print(f"Training time: {time() - start_time:.2f} seconds")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return acc


def compare_methods(hc_acc, pca_acc):
    """Compare the performance of both methods visually."""
    plt.figure(figsize=(8, 5))
    plt.bar(['Hierarchical Clustering', 'PCA'], [hc_acc, pca_acc])
    plt.ylabel('Accuracy')
    plt.title('Comparison of Feature Reduction Methods')
    plt.ylim(0, 1)
    plt.show()


def main():
    """Main function to execute the complete workflow."""
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, le = load_and_preprocess_data()

    # Step 2: Hierarchical clustering feature selection
    print("\nPerforming hierarchical clustering feature selection...")
    selected_features_hc = hierarchical_feature_selection(X_train, y_train)
    X_train_hc = X_train[:, selected_features_hc]
    X_test_hc = X_test[:, selected_features_hc]
    print(f"Shape after hierarchical clustering feature selection: {X_train_hc.shape}")

    # Step 3: PCA feature reduction
    print("\nPerforming PCA dimensionality reduction...")
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)
    print(f"Shape after PCA: {X_train_pca.shape}")

    # Step 4: Train and evaluate models
    hc_acc = train_and_evaluate(X_train_hc, X_test_hc, y_train, y_test,
                                "Hierarchical Clustering Feature Selection", le)
    pca_acc = train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test,
                                 "PCA Feature Reduction", le)

    # Step 5: Compare methods
    compare_methods(hc_acc, pca_acc)


if __name__ == "__main__":
    main()