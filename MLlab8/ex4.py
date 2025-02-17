import numpy as np
import pandas as pd


# Ordinal Encoding
def ordinal_encode(categories, data):
    category_map = {category: index for index, category in enumerate(categories)}

    # Handle NaN values by replacing them with a valid category or removing them
    data = [item if not pd.isna(item) else 'Unknown' for item in data]  # Replace NaN with 'Unknown'

    # Handle 'Unknown' by assigning a special index (e.g., -1) if it's not in the category_map
    encoded_data = []
    for item in data:
        if item in category_map:
            encoded_data.append(category_map[item])
        else:
            encoded_data.append(-1)  # You can assign a default value like -1 for 'Unknown'
    return encoded_data


# One-Hot Encoding
def one_hot_encode(categories, data):
    category_map = {category: index for index, category in enumerate(categories)}

    # Handle NaN values the same way as in ordinal encoding
    data = [item if not pd.isna(item) else 'Unknown' for item in data]  # Replace NaN with 'Unknown'

    one_hot_encoded_data = []
    for item in data:
        # Handle 'Unknown' by adding a vector of 0s or some default encoding
        if item in category_map:
            one_hot_vector = [0] * len(categories)
            one_hot_vector[category_map[item]] = 1
        else:
            one_hot_vector = [0] * len(categories)  # Or a default vector for unknowns (e.g., all 0s)
        one_hot_encoded_data.append(one_hot_vector)

    return one_hot_encoded_data


# Function to load the breast cancer dataset
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
    columns = ['id', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
               'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
               'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
    df = pd.read_csv(url, names=columns)

    # Handling missing values (replace '?' with NaN and then fill with the mode or median)
    df.replace('?', np.nan, inplace=True)
    df['Bare_Nuclei'] = df['Bare_Nuclei'].fillna(df['Bare_Nuclei'].mode()[0])
    df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')

    return df


# Load data
df = load_data()

# Print the first few rows of the data
print("First few rows of the dataset:")
print(df.head())

# Ordinal Encoding example (for the 'Class' column with 'M' and 'B')
categories = ['M', 'B']  # Malignant and Benign
data = df['Class'].values  # Data to be encoded

# Ordinal Encoding
ordinal_encoded_data = ordinal_encode(categories, data)
print("\nOrdinal Encoding (Class):", ordinal_encoded_data[:10])  # Show first 10 entries

# One-Hot Encoding example (for the 'Class' column)
one_hot_encoded_data = one_hot_encode(categories, data)
print("\nOne-Hot Encoding (Class):")
for vector in one_hot_encoded_data[:10]:  # Show first 10 vectors
    print(vector)
