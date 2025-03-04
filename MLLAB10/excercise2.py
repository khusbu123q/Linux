import numpy as np
import pandas as pd
from collections import Counter


def calculate_entropy(labels):

    label_counts = Counter(labels)
    total_samples = len(labels)

    entropy = 0.0
    for count in label_counts.values():
        probability = count / total_samples
        entropy -= probability * np.log2(probability)

    return entropy


def information_gain(parent_labels, child_labels_1, child_labels_2):

    parent_entropy = calculate_entropy(parent_labels)

    total_samples = len(parent_labels)
    weight_1 = len(child_labels_1) / total_samples
    weight_2 = len(child_labels_2) / total_samples

    children_entropy = (weight_1 * calculate_entropy(child_labels_1)) + (weight_2 * calculate_entropy(child_labels_2))

    return parent_entropy - children_entropy

def main():
    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')


    data_labels = df['disease_score'].tolist()


    child_1_labels = data_labels[:len(data_labels) // 2]
    child_2_labels = data_labels[len(data_labels) // 2:]

    info_gain = information_gain(data_labels, child_1_labels, child_2_labels)
    print(f"Information Gain: {info_gain:.4f}")


if __name__ =="__main__":
    main()