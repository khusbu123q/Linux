import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def l2_norm(theta):
    return math.sqrt(sum(v**2 for v in theta))


def l1_norm(theta):
    return sum(abs(v) for v in theta)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

data = load_breast_cancer()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


ridge_clf = RidgeClassifier(alpha=1.0)
ridge_clf.fit(X_train, y_train)


lasso_clf = Lasso(alpha=0.1)
lasso_clf.fit(X_train, y_train)


ridge_pred = ridge_clf.predict(X_test)
lasso_pred = lasso_clf.predict(X_test)


print("Ridge Classifier Evaluation:")
print(classification_report(y_test, ridge_pred))
print("Confusion Matrix for Ridge Classifier:")
print(confusion_matrix(y_test, ridge_pred))


print("\nLasso Classifier Evaluation:")

lasso_pred_thresholded = np.round(lasso_pred)
print(classification_report(y_test, lasso_pred_thresholded))
print("Confusion Matrix for Lasso Classifier:")
print(confusion_matrix(y_test, lasso_pred_thresholded))


plt.figure(figsize=(10, 6))


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
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
#
# # Load the Wisconsin Breast Cancer dataset
# data = load_breast_cancer()
# X = data.data
# y = data.target
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
#
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # Train the Logistic Regression model (without regularization)
# log_reg_clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
# log_reg_clf.fit(X_train, y_train)
#
# # Make predictions
# log_reg_pred = log_reg_clf.predict(X_test)
#
# # Print the classification evaluation metrics
# print("Logistic Regression (No Regularization) Evaluation:")
# print(classification_report(y_test, log_reg_pred))
# print("Confusion Matrix for Logistic Regression (No Regularization):")
# print(confusion_matrix(y_test, log_reg_pred))
#
# # Plot the coefficients of the model
# plt.figure(figsize=(10, 6))
# plt.barh(range(len(log_reg_clf.coef_[0])), log_reg_clf.coef_[0])
# plt.title('Logistic Regression Coefficients (No Regularization)')
# plt.xlabel('Coefficient Value')
# plt.ylabel('Feature Index')
# plt.tight_layout()
# plt.show()
