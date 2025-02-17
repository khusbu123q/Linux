import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.datasets import fetch_california_housing


def load_california_housing():
    california_housing = fetch_california_housing()
    df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    df['target'] = california_housing.target
    return df


df = load_california_housing()


print("First few rows of the California Housing dataset:")
print(df.head())


ocean_proximity_categories = ['NEAR BAY', 'NEAR OCEAN', 'INLAND']
repeated_ocean_proximity = np.tile(ocean_proximity_categories, len(df) // len(ocean_proximity_categories))


df['Ocean_Proximity'] = repeated_ocean_proximity[:len(df)]


print("\nDataset with simulated 'Ocean_Proximity' column:")
print(df[['Ocean_Proximity']].head())


ordinal_encoder = OrdinalEncoder(categories=[['INLAND', 'NEAR OCEAN', 'NEAR BAY']])
ocean_proximity_data = df[['Ocean_Proximity']].values

ordinal_encoded_data = ordinal_encoder.fit_transform(ocean_proximity_data)


print("\nOrdinal Encoding (Ocean_Proximity):")
print(ordinal_encoded_data[:10])


one_hot_encoder = OneHotEncoder(sparse_output=False)


one_hot_encoded_data = one_hot_encoder.fit_transform(ocean_proximity_data)


print("\nOne-Hot Encoding (Ocean_Proximity):")
print(one_hot_encoded_data[:10])