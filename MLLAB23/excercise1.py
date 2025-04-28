import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_and_prepare_data():
    """Load and prepare the Iris dataset with noise and discretization."""
    # Load Iris dataset
    iris = load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])

    # Use only first two features
    features = ['sepal length (cm)', 'sepal width (cm)']
    X = data[features]
    y = data['target']

    # Add random noise
    np.random.seed(42)
    X_noisy = X + np.random.normal(0, 0.1, size=X.shape)

    # Discretize features (5 bins for each feature)
    X_discrete = X_noisy.apply(lambda x: pd.cut(x, bins=5, labels=False))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_discrete, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def train_joint_prob_model(X_train, y_train):
    """
    Train a joint probability model.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        DataFrame with joint probabilities
    """
    # Create a DataFrame with features and target
    train_data = X_train.copy()
    train_data['target'] = y_train
    # Calculate joint probabilities
    joint_prob = train_data.groupby(
        ['sepal length (cm)', 'sepal width (cm)', 'target']
    ).size().reset_index(name='count')

    total = joint_prob['count'].sum()
    joint_prob['probability'] = joint_prob['count'] / total

    return joint_prob


def predict_with_joint_prob(model, X_test):
    """
    Make predictions using the joint probability model.

    Args:
        model: Trained joint probability model
        X_test: Test features

    Returns:
        Array of predictions
    """
    predictions = []
    for _, row in X_test.iterrows():
        sepal_len = row['sepal length (cm)']
        sepal_wid = row['sepal width (cm)']

        # Find matching probabilities
        matches = model[
            (model['sepal length (cm)'] == sepal_len) &
            (model['sepal width (cm)'] == sepal_wid)
            ]

        if len(matches) == 0:
            # If no exact match, predict the most common class in training
            predicted_class = model['target'].mode()[0]
        else:
            # Predict class with highest probability
            predicted_class = matches.loc[matches['probability'].idxmax(), 'target']

        predictions.append(predicted_class)

    return np.array(predictions)


def train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a decision tree model.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels

    Returns:
        Accuracy score of the decision tree
    """
    # Train decision tree with max_depth=2
    dt_model = DecisionTreeClassifier(max_depth=2, random_state=42)
    dt_model.fit(X_train, y_train)

    # Evaluate decision tree
    y_pred_dt = dt_model.predict(X_test)
    return accuracy_score(y_test, y_pred_dt)


def main():
    """Main function to run the comparison."""
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Train and evaluate joint probability model
    joint_prob_model = train_joint_prob_model(X_train, y_train)
    y_pred_joint = predict_with_joint_prob(joint_prob_model, X_test)
    joint_accuracy = accuracy_score(y_test, y_pred_joint)

    # Train and evaluate decision tree
    dt_accuracy = train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test)

    # Print results
    print("\nModel Comparison Results:")
    print(f"Joint Probability Model Accuracy: {joint_accuracy:.4f}")
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

    if joint_accuracy > dt_accuracy:
        print("Joint probability model performed better")
    elif dt_accuracy > joint_accuracy:
        print("Decision tree performed better")
    else:
        print("Both models performed equally")

    # Show sample of the joint probability model
    print("\nSample of Joint Probability Model:")
    print(joint_prob_model.head())


if __name__ == "__main__":
    main()