
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("/content/drive/MyDrive/csv/PhiUSIIL_Phishing_URL_Dataset.csv")

df.head()

df.info()

display(df['Crypto'].value_counts())

print(f"Number of null values in 'LetterRatioInURL': {df['LetterRatioInURL'].isnull().sum()}")

print(f"Number of null values in 'IsDomainIP': {df['IsDomainIP'].isnull().sum()}")

df.columns

df.dropna(inplace=True)
print("Rows with null values have been dropped. Now we have a processed dataset.")

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'TLD' column
df['TLD_encoded'] = label_encoder.fit_transform(df['TLD'])

# Display the first few rows with the new encoded column
display(df.head())

tld_counts = df['TLD'].value_counts()
print(tld_counts)

# Define the list of top TLDs to keep
top_tlds = ['com', 'org', 'net', 'app', 'uk']

# Filter the DataFrame to keep only rows where the 'tld' is in the top_tlds list
df = df[df['TLD'].isin(top_tlds)]

# Display the first few rows of the filtered DataFrame
display(df.head())

df['TLD_encoded'] = label_encoder.fit_transform(df['TLD'])


display(df.head())

df.info()

df['label'].unique()

label_counts = df['label'].value_counts()
print(label_counts)

df.describe()

df.tail()

# Calculate the correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap of Features')
plt.show()

# Calculate absolute correlations with the 'label' column
abs_correlations = correlation_matrix['label'].abs().sort_values(ascending=False)

# Select the top 12 highly correlated columns (excluding 'label' itself)
top_25_features = abs_correlations[1:26].index.tolist()

# Include 'label' in the list of columns to keep
columns_to_keep = top_25_features + ['label']

print("Columns to keep:", columns_to_keep)

# Drop the other columns from the DataFrame
df = df[columns_to_keep]

print("DataFrame updated with the top 25 features most correlated with 'label'.")
display(df.head())

print(df['label'].value_counts())

df.describe()

for column in df.columns:
    if column != 'label':
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Box plot of {column}')
        plt.show()