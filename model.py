# --- 1. Import Necessary Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better readability of plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- 2. Load and Initial Data Inspection ---
print("--- Loading Data ---")
try:
    df = pd.read_csv('StressLevelDataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'StressLevelDataset.csv' not found. Ensure it is in the same directory.")
    exit()

print("\n--- First 5 rows of the dataset: ---")
print(df.head())

print("\n--- Dataset Information: ---")
df.info()

print("\n--- Key Statistical Metrics: ---")
print(df.describe())
# Conclusion: All columns are non-null and numeric (int64), confirming data quality.

# --- 3. Target Variable Analysis (stress_level) ---
print("\n--- Analyzing the Target Variable 'stress_level' ---")
stress_counts = df['stress_level'].value_counts()
print("Class Distribution (Stress Levels):")
print(stress_counts)
# Conclusion: The classes are reasonably balanced, which is good for model training.

# Visualize stress level distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='stress_level', data=df, palette='viridis')
plt.title('Distribution of Student Stress Levels')
plt.xlabel('Stress Level (0: Low, 1: Medium, 2: High)')
plt.ylabel('Number of Students')
plt.show()


# --- 4. Correlation Analysis ---
# Goal: Find features most strongly correlated with the stress level.
print("\n--- Generating Correlation Heatmap ---")
plt.figure(figsize=(18, 15))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Analyze correlations with the target variable
stress_correlation = correlation_matrix['stress_level'].sort_values(ascending=False)
print("\n--- Feature Correlation with Stress Level: ---")
print(stress_correlation)
# Conclusion: 'anxiety_level' and 'depression' are among the strongest positive correlates.
# 'self_esteem' and 'sleep_quality' are among the strongest negative correlates.

# --- 5. Visual Analysis of Key Factors ---
# Goal: Visually confirm how feature values change across different stress levels.

# Select key features based on correlation and project relevance
key_features = [
    'anxiety_level', 
    'depression',
    'self_esteem',
    'sleep_quality', 
    'academic_performance',
    'social_support'
]

print(f"\n--- Visualizing the Impact of Key Factors on Stress Level ---")

for feature in key_features:
    plt.figure(figsize=(8, 6))
    # Using 'magma' palette which works well for boxplots
    sns.boxplot(x='stress_level', y=feature, data=df, palette='magma')
    plt.title(f'Impact of "{feature}" on Stress Level')
    plt.xlabel('Stress Level')
    plt.ylabel(f'Value of "{feature}"')
    plt.show()

print("\n--- Exploratory Data Analysis complete. ---")
print("Insights are ready for inclusion in the project report.")