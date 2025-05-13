# Analyzing Data with Pandas and Visualizing Results with Matplotlib

# Objective:
# 1. Load and analyze a dataset using pandas
# 2. Create visualizations using matplotlib and seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
print("Loading Iris dataset...")
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData types:")
print(df.dtypes)

print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Since there are no missing values, no cleaning is necessary

# Task 2: Basic Data Analysis

print("\nDescriptive statistics:")
print(df.describe())

# Grouping by species and calculating the mean of each feature
print("\nAverage values per species:")
print(df.groupby('species').mean())

# Interesting finding
print("\nObservation: Setosa typically has smaller petal length and width than other species.")

# Task 3: Data Visualization

# Line plot - mean feature values per species
plt.figure(figsize=(8, 5))
df.groupby('species').mean().T.plot(marker='o')
plt.title('Average Feature Values per Species')
plt.ylabel('Mean value')
plt.xlabel('Features')
plt.grid(True)
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('line_plot.png')
plt.show()

# Bar chart - average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x='species', y='petal length (cm)', data=df, estimator='mean')
plt.title('Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.show()

# Histogram - distribution of sepal length
plt.figure(figsize=(6, 4))
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram.png')
plt.show()

# Scatter plot - sepal length vs petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.show()

# Error handling example (uncomment to test)
# try:
#     df = pd.read_csv("nonexistent.csv")
# except FileNotFoundError:
#     print("Error: File not found.")

print("\nScript execution complete. Visualizations saved as PNG files.")
