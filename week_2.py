import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Load the logistics dataset
df = pd.read_csv("dataset.csv")

print("Original Dataset:")
print(df.head())

# 2. Display basic information
print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

# 3. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Remove duplicate records
duplicates = df.duplicated().sum()
print("\nNumber of Duplicate Records:", duplicates)

df = df.drop_duplicates()

# 5. Display statistical information
print("\nStatistical Summary:")
print(df.describe())

# 6. Handle missing numerical values
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

# 7. Detect outliers using IQR
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print("\nColumn:", column)
    print("Number of Outliers:", len(outliers))

# 8. Normalize numerical data
scaler = MinMaxScaler()

df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 9. Check the processed dataset
print("\nProcessed Dataset:")
print(df.head())

# 10. Check missing values after preprocessing
print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

# 11. Save the cleaned dataset
df.to_csv("cleaned_logistics_data.csv", index=False)

print("\nPreprocessing completed successfully!")
print("Cleaned dataset saved as cleaned_logistics_data.csv")
