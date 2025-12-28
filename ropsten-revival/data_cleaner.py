import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns=None, outlier_threshold=1.5):
    """
    Main cleaning function for datasets
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    original_shape = df.shape
    cleaned_df = df.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
            removal_stats[col] = removed
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Removed outliers per column: {removal_stats}")
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist and have no null values
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        print(f"Warning: Found null values in columns:\n{null_counts[null_counts > 0]}")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    })
    
    cleaned_data = clean_dataset(sample_data)
    print("Data cleaning completed successfully")