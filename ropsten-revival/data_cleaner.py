
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def clean_dataset(df, numeric_columns):
    original_shape = df.shape
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
    print(f"Original shape: {original_shape}, Cleaned shape: {df.shape}")
    print(f"Removed {original_shape[0] - df.shape[0]} rows")
    return df