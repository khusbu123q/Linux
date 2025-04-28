import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load CIFAR batch
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        X = dict[b'data']
        Y = dict[b'labels']
        return X, Y

# Load full CIFAR-10 dataset
def load_cifar10_data(data_dir):
    X_train, y_train = [], []
    for i in range(1, 6):
        file = os.path.join(data_dir, f'data_batch_{i}')
        X_batch, y_batch = load_cifar_batch(file)
        X_train.append(X_batch)
        y_train.extend(y_batch)
    X_train = np.concatenate(X_train)
    y_train = np.array(y_train)
    X_test, y_test = load_cifar_batch(os.path.join(data_dir, 'test_batch'))
    return X_train, y_train, np.array(X_test), np.array(y_test)

def preprocess(X):
    return X.astype("float32") / 255.0

def show_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def show_predictions(X, y_true, y_pred, class_names, n=10, apply_pca=False, pca=None):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        if apply_pca:
            # Reverse PCA transformation to get back original image data
            img = pca.inverse_transform(X[i]).reshape(3, 32, 32).transpose(1, 2, 0)
            # Clip values to valid range [0, 1] for displaying images
            img = np.clip(img, 0, 1)
        else:
            # If no PCA applied, reshape back to 3, 32, 32 image
            img = X[i].reshape(3, 32, 32).transpose(1, 2, 0)
            # Clip values to valid range [0, 1] for displaying images
            img = np.clip(img, 0, 1)

        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(f"T: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]}", fontsize=9)
        plt.axis('off')
    plt.suptitle("Sample Predictions")

    plt.show()


# Main
data_dir = "/home/ibab/Downloads/cifar-10-batches-py"
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)
X_train = preprocess(X_train)
X_test = preprocess(X_test)

# Reduce dataset for faster computation
num_train, num_test = 10000, 2000
X_train, y_train = X_train[:num_train], y_train[:num_train]
X_test, y_test = X_test[:num_test], y_test[:num_test]

# Optional: PCA for speed-up
apply_pca = True
if apply_pca:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=100)  # reduce from 3072 to 100
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

# Train kNN
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"kNN CIFAR-10 Accuracy: {acc:.4f}")

# Plot confusion matrix and sample predictions
show_confusion_matrix(y_test, y_pred, class_names)
# Show predictions (with PCA applied)
show_predictions(X_test[:10], y_test[:10], y_pred[:10], class_names, apply_pca=True, pca=pca)

