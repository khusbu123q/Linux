import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt  # Corrected import

# Initialize SHAP JS visualizations
shap.initjs()

# Load the Adult Census Income dataset
X_full, y_full = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=7)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set model hyperparameters
xgb_params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss",
}

# Train the XGBoost model
booster_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=5000,
    evals=[(dtest, "Test")],
    verbose_eval=100,
    early_stopping_rounds=20,
)

# Feature importance plots
xgb.plot_importance(booster_model)
plt.title("Feature Importance (Weight)")
plt.show()

xgb.plot_importance(booster_model, importance_type="cover")
plt.title("Feature Importance (Cover)")
plt.show()

xgb.plot_importance(booster_model, importance_type="gain")
plt.title("Feature Importance (Gain)")
plt.show()

# SHAP explainability analysis
print("Calculating SHAP values... (This might take some time)")

explainer = shap.TreeExplainer(booster_model)
shap_values = explainer.shap_values(X_full)

# SHAP force plot for a single prediction
shap.force_plot(explainer.expected_value, shap_values[0, :], X_display.iloc[0, :])

# SHAP force plot for first 1000 samples
shap.force_plot(explainer.expected_value, shap_values[:1000, :], X_display.iloc[:1000, :])

# SHAP summary plots
shap.summary_plot(shap_values, X_display, plot_type="bar")  # Feature importance bar plot
shap.summary_plot(shap_values, X_display)  # SHAP summary beeswarm plot
