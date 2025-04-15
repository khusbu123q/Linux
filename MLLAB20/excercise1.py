
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ISLP import load_data
from statsmodels.datasets import get_rdataset


def boxplot(score, coeff, labels=None, states=None):
    xs = score[:, 0]
    ys = score[:, 1]
    zs = score[:, 2]
    n = coeff.shape[0]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, s=5)

    if states is not None:
        for i in range(len(xs)):
            ax.text(xs[i], ys[i], zs[i], states[i], size=7)

    for i in range(n):
        ax.quiver(0, 0, 0, coeff[i, 0], coeff[i, 1], coeff[i, 2], color='r', alpha=0.5)
        label = labels[i] if labels is not None else f"Var{i + 1}"
        ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, coeff[i, 2] * 1.15, label, color='g', ha='center', va='center')

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("3D Boxplot")
    plt.grid()
    plt.show()


def main():
    # Load data
    df = get_rdataset('USArrests').data
    X = df.values.squeeze()
    states = df.index
    corr_df = df.corr()
    labels = corr_df.columns
    # data standardization
    X_std = StandardScaler().fit_transform(X)
    pca = PCA()
    X_std_trans = pca.fit_transform(X_std)
    df_std_pca = pd.DataFrame(X_std_trans)
    std = df_std_pca.describe().transpose()["std"]
    print(f"Standard deviation: {std.values}")
    print(f"Proportion of Variance Explained: {pca.explained_variance_ratio_}")
    print(f"Cumulative Proportion: {np.cumsum(pca.explained_variance_)}")

    # 3D biplot
    boxplot(X_std_trans[:, 0:3], np.transpose(pca.components_[0:3, :]), labels=list(labels), states=states)

    ### From this boxplot, we see that Assault and UrbanPop are the most important features as the arrows to each of these dominate the boxplot.

    # Feature importance for the first 3 principal components
    pc1 = abs(pca.components_[0])
    pc2 = abs(pca.components_[1])
    pc3 = abs(pca.components_[2])
    feat_df = pd.DataFrame()
    feat_df["Features"] = list(labels)
    feat_df["PC1 Importance"] = pc1
    feat_df["PC2 Importance"] = pc2
    feat_df["PC3 Importance"] = pc3
    print(feat_df)

    # Inspecting the feature importance now, it seems that most of the variables contribute fairly evenly, with only some with low importance.

    # Cumulative variance plot
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_),
             c='red')
    plt.title("Cumulative Explained Variance")
    plt.show()

    # Scree plot
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title("Scree plot")
    plt.show()

    ## From the plots above, it seems the first 3 principal components together explain around 95% of the variance. We can therefore use them to perform model training.

    # PCA dataset creation:
    pca_df = pd.DataFrame(X_std_trans[:, 0:3], index=df.index)
    print(pca_df.head())


if __name__ == "__main__":
    main()

