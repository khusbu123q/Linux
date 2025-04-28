import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv("spam_sms.csv")
        df = df[['v1', 'v2']]  # Select only the label and text columns
        df.columns = ['label', 'text']  # Rename columns
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary
        return df
    except FileNotFoundError:
        print(f"Error: File  not found. Please download the dataset from Kaggle.")
        return None


def explore_data(df):
    """Display basic information about the dataset."""
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nPercentage of spam messages: {df['label'].mean():.2%}")


def train_model(X_train, y_train):
    """Train the Naive Bayes classifier."""
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    return classifier, vectorizer


def evaluate_model(classifier, vectorizer, X_test, y_test):
    """Evaluate the model performance."""
    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualize confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return y_pred


def predict_sample_messages(classifier, vectorizer, messages):
    """Make predictions on sample messages."""
    sample_vec = vectorizer.transform(messages)
    predictions = classifier.predict(sample_vec)
    probabilities = classifier.predict_proba(sample_vec)

    print("\nSample Message Predictions:")
    for msg, pred, prob in zip(messages, predictions, probabilities):
        print(f"\nMessage: {msg}")
        print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}")
        print(f"Probability: Ham - {prob[0]:.4f}, Spam - {prob[1]:.4f}")


def main():
    """Main function to run the spam detection pipeline."""
    # Load and prepare data
    df = load_data()
    if df is None:
        return

    explore_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # Train model
    classifier, vectorizer = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(classifier, vectorizer, X_test, y_test)

    # Test with sample messages
    sample_messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)",
        "Hey, are we still meeting for lunch tomorrow?",
        "Congratulations! You've been selected for a free iPhone. Click here to claim now!",
        "Hi Mom, I'll be home late tonight. Don't wait up for me.",
        "URGENT! Your bank account has been compromised. Click to secure your account now!"
    ]

    predict_sample_messages(classifier, vectorizer, sample_messages)


if __name__ == "__main__":
    main()