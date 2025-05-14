# Task 1: Load and Explore the Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!\n")
except Exception as e:
    print("Error loading dataset:", e)

# Display the first few rows
print("First five rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Clean the dataset (no missing values in Iris, but here's how you’d drop them)
df.dropna(inplace=True)

# Task 2: Basic Data Analysis

# Descriptive statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and calculate mean of numerical features
print("\nMean of numerical features by species:")
print(df.groupby('species').mean())

# Task 3: Data Visualization

# Set Seaborn style
sns.set(style="whitegrid")

# Line Chart: Mean measurements for each species (not time-series but line plot of mean values)
species_means = df.groupby('species').mean().T
species_means.plot(kind='line', marker='o')
plt.title("Average Feature Measurements by Species")
plt.xlabel("Features")
plt.ylabel("Mean Measurement (cm)")
plt.legend(title="Species")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar Chart: Average petal length per species
sns.barplot(data=df, x='species', y='petal length (cm)', ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram: Distribution of sepal length
sns.histplot(data=df, x='sepal length (cm)', kde=True)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Sepal length vs Petal length
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
