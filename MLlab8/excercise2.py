# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Preprocess the data (standardize features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 4: Create and train the Ridge classifier
ridge_classifier = RidgeClassifier()
ridge_classifier.fit(X_train, y_train)

# Step 5: Create and train the Lasso classifier (using LogisticRegression with L1 penalty)
lasso_classifier = LogisticRegression(penalty='l1', solver='liblinear')
lasso_classifier.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the models
ridge_predictions = ridge_classifier.predict(X_test)
lasso_predictions = lasso_classifier.predict(X_test)


ridge_accuracy = accuracy_score(y_test, ridge_predictions)
lasso_accuracy = accuracy_score(y_test, lasso_predictions)

# Step 7: Output the results
print(f"Ridge classifier accuracy: {ridge_accuracy:.4f}")
print(f"Lasso classifier accuracy: {lasso_accuracy:.4f}")

