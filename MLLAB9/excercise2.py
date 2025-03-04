import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')


label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])


X = df[["age", "BMI", "BP", "blood_sugar", "Gender"]].values
y = df["disease_score"].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


regressor = DecisionTreeRegressor(max_depth=4)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Feature importance visualization
feature_names = ["age", "BMI", "BP", "blood_sugar", "Gender"]
importances = regressor.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Decision Tree Regression")
plt.show()
