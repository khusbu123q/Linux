import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
X = df[["age","BMI","BP","blood_sugar","Gender"]]
y = df["disease_score_fluct"]

print(df.head)
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())


# histogram plot
df.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.show()

# for box plots of each column
for i in df.columns:
    sns.boxplot(y=df[i])
    plt.show()

# pairplot to see correlation between different columns
sns.pairplot(df,diag_kind="kde")
plt.show()

#correlation and heatmap
print(df.corr())
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()


