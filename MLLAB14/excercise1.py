# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def load_data():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y

def main():
    X, y = load_data()

    # Define K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 folds

    # Create a DecisionTreeClassifier as the base estimator
    base_estimator = DecisionTreeClassifier(max_depth=1)

    # Initialize variables to store results
    accuracy_scores = []
    reports = []

    # K-Fold Cross-Validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Instantiate AdaBoost Classifier with 50 weak learners
        adaboost_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)

        # Train the model
        adaboost_clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = adaboost_clf.predict(X_test)

        # Evaluate model performance
        accuracy_scores.append(accuracy_score(y_test, y_pred))

        # Generate classification report as a dictionary
        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(report)

    # Print average accuracy across all folds
    avg_accuracy = np.mean(accuracy_scores)
    print(f"Average Accuracy across {kf.n_splits} folds: {avg_accuracy:.2f}")

    # Summarize classification report across folds
    avg_report = {}

    # Process only the class labels (0, 1, 2) and exclude avg/accuracy fields
    for key in reports[0].keys():
        if isinstance(reports[0][key], dict):  # Skip accuracy and avg fields
            avg_report[key] = {
                'precision': np.mean([report[key]['precision'] for report in reports if key in report]),
                'recall': np.mean([report[key]['recall'] for report in reports if key in report]),
                'f1-score': np.mean([report[key]['f1-score'] for report in reports if key in report])
            }

    # Print summarized classification report
    for label, metrics in avg_report.items():
        print(f"\nClass {label}:")
        print(f"  Precision: {metrics['precision']:.2f}")
        print(f"  Recall:    {metrics['recall']:.2f}")
        print(f"  F1-Score:  {metrics['f1-score']:.2f}")

if __name__ == "__main__":
    main()
