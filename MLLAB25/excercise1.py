import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X_full, y_full = shap.datasets.adult()
    X_display, y_display = shap.datasets.adult(display=True)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=7)
    return X_full, y_full, X_display, y_display, X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_params = {
        "eta": 0.01,
        "objective": "binary:logistic",
        "subsample": 0.5,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss",
    }

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtest, "Test")],
        verbose_eval=100,
        early_stopping_rounds=20,
    )

    return model

def plot_feature_importance(model):

    plt.figure()
    xgb.plot_importance(model)
    plt.title("Feature Importance (Weight)")
    plt.show()

    plt.figure()
    xgb.plot_importance(model, importance_type="cover")
    plt.title("Feature Importance (Cover)")
    plt.show()

    plt.figure()
    xgb.plot_importance(model, importance_type="gain")
    plt.title("Feature Importance (Gain)")
    plt.show()

def compute_shap_values(model, X_full):
    print("Calculating SHAP values... ")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_full)
    return explainer, shap_values

def plot_shap_explanations(explainer, shap_values, X_display):

    # Force plot for a single prediction
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_display.iloc[0, :])

    # Force plot for first 1000 samples
    shap.force_plot(explainer.expected_value, shap_values[:1000, :], X_display.iloc[:1000, :])

    # SHAP summary plots
    shap.summary_plot(shap_values, X_display, plot_type="bar")  # Feature importance bar plot
    shap.summary_plot(shap_values, X_display)                   # Beeswarm plot

def main():
    shap.initjs()

    # Step 1: Load data
    X_full, y_full, X_display, y_display, X_train, X_test, y_train, y_test = load_data()

    # Step 2: Train model
    model = train_model(X_train, y_train, X_test, y_test)

    # Step 3: Plot feature importances
    plot_feature_importance(model)

    # Step 4: Compute SHAP values
    explainer, shap_values = compute_shap_values(model, X_full)

    # Step 5: Plot SHAP explanations
    plot_shap_explanations(explainer, shap_values, X_display)

if __name__ == "__main__":
    main()
