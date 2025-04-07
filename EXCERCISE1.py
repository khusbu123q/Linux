import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('heart.csv')
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# **Correction:** Verify the target variable to ensure it's binary
print("Unique values in the 'target' column:", df['output'].unique())
print("Value counts in the 'target' column:\n", df['output'].value_counts())

# Separate features (X) and target (y)
X = df.drop('sex', axis=1)
y = df['output']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Get the probability predictions for the positive class (heart disease)
y_prob = model.predict_proba(X_test)[:, 1]

# Define a range of thresholds to vary
thresholds = np.arange(0.1, 1.0, 0.1)

# Store metrics for each threshold
metrics = []

# Generate confusion matrices and calculate metrics for each threshold
print("\nConfusion Matrices and Metrics for Different Thresholds:")
for threshold in thresholds:
    # Predict class labels based on the current threshold
    y_pred_threshold = (y_prob >= threshold).astype(int)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred_threshold)

    # **Correction:** Ensure the confusion matrix is for binary classification (2x2)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_threshold)
        precision = precision_score(y_test, y_pred_threshold, zero_division=0)  # Handle potential division by zero
        sensitivity = recall_score(y_test, y_pred_threshold, zero_division=0) # Recall is the same as sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred_threshold, zero_division=0)

        metrics.append({
            'threshold': threshold,
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1
        })

        print(f"\nThreshold: {threshold:.2f}")
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1-score: {f1:.4f}")
    else:
        print(f"\nWarning: Confusion matrix for threshold {threshold:.2f} is not 2x2. Skipping metric calculation.")
        print("Confusion Matrix:")
        print(cm)

 #Plot the ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity = 1 - FPR)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"\nArea Under the ROC Curve (AUC): {roc_auc:.4f}")