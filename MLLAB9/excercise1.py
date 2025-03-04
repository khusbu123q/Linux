import numpy as np
import pandas as pd

# Simulate dataset
np.random.seed(42)
data_size = 200
age = np.random.randint(18, 70, size=data_size)
BMI = np.random.uniform(18.5, 35, size=data_size)
BP = np.random.randint(70, 150, size=data_size)
blood_sugar = np.random.randint(70, 200, size=data_size)
disease_score = np.random.uniform(0, 1, size=data_size)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'BMI': BMI,
    'BP': BP,
    'blood_sugar': blood_sugar,
    'disease_score': disease_score
})

def partition_dataset(df, threshold):
    """Partition dataset into two parts based on BP threshold."""
    df_low = df[df['BP'] <= threshold]
    df_high = df[df['BP'] > threshold]
    return df_low, df_high

# Partition based on different threshold values
thresholds = [80, 78, 82]
partitioned_data = {}

for t in thresholds:
    df_low, df_high = partition_dataset(df, t)
    partitioned_data[f't_{t}_low'] = df_low
    partitioned_data[f't_{t}_high'] = df_high

# Display partitioned dataset sizes
for key, data in partitioned_data.items():
    print(f"Dataset '{key}': {data.shape[0]} rows")
