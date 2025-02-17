# Import necessary libraries
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np


sonar = fetch_openml('sonar', version=1)


X = sonar.data
y = sonar.target


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


log_reg = LogisticRegression(max_iter=10000)


cv_scores = cross_val_score(log_reg, X_scaled, y, cv=10, scoring='accuracy')


print(f"Cross-validation scores for each fold: {cv_scores}")
print(f"Mean cross-validation score: {np.mean(cv_scores)}")
print(f"Standard deviation of cross-validation scores: {np.std(cv_scores)}")
