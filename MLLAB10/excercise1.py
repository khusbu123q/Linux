import numpy as np
from collections import Counter
import pandas as pd


def calculate_entropy(labels):

    label_counts = Counter(labels)
    total_samples = len(labels)

    entropy = 0.0
    for count in label_counts.values():
        probability = count / total_samples
        entropy -= probability * np.log2(probability)

    return entropy


# # Example usage
# data_labels = ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
# entropy_value = calculate_entropy(data_labels)
# print(f"Entropy: {entropy_value:.4f}")
def main():
# Load dataset
    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')


    data_labels = df['disease_score'].tolist()
    entropy_value = calculate_entropy(data_labels)
    print(f"Entropy: {entropy_value:.4f}")

if __name__=="__main__":
    main()