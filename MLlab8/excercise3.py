import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, r2_score

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler (optional but recommended for regularization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Ridge Classifier (L2 regularization)
ridge_clf = RidgeClassifier(alpha=1.0)  # alpha is the regularization strength
ridge_clf.fit(X_train, y_train)

# Train the Lasso Classifier (L1 regularization)
lasso_clf = Lasso(alpha=0.1)
lasso_clf.fit(X_train, y_train)

# Make predictions
ridge_pred = ridge_clf.predict(X_test)
lasso_pred = lasso_clf.predict(X_test)

# For Lasso, the output is continuous, so we round it to get binary predictions
lasso_pred_thresholded = np.round(lasso_pred)

# Print the classification evaluation metrics
print("Ridge Classifier Evaluation:")
print(classification_report(y_test, ridge_pred))
print("Confusion Matrix for Ridge Classifier:")
print(confusion_matrix(y_test, ridge_pred))

# Print Lasso evaluation
print("\nLasso Classifier Evaluation:")
print(classification_report(y_test, lasso_pred_thresholded))
print("Confusion Matrix for Lasso Classifier:")
print(confusion_matrix(y_test, lasso_pred_thresholded))

# Calculate R² score for Ridge Classifier
ridge_r2 = r2_score(y_test, ridge_pred)
print(f"R² Score for Ridge Classifier: {ridge_r2:.4f}")

# Calculate R² score for Lasso Classifier (after thresholding the predictions)
lasso_r2 = r2_score(y_test, lasso_pred_thresholded)
print(f"R² Score for Lasso Classifier: {lasso_r2:.4f}")

# Plot the coefficients for both models
plt.figure(figsize=(10, 6))

# Ridge Classifier coefficients
plt.subplot(1, 2, 1)
plt.barh(range(len(ridge_clf.coef_)), ridge_clf.coef_)
plt.title('Ridge Classifier Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Index')


plt.subplot(1, 2, 2)
plt.barh(range(len(lasso_clf.coef_)), lasso_clf.coef_)
plt.title('Lasso Classifier Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Index')

plt.tight_layout()
plt.show()
