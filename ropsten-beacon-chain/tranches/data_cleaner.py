import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    """Normalize specified column to range [0,1]."""
    if column_name in df.columns:
        col_min = df[column_name].min()
        col_max = df[column_name].max()
        if col_max != col_min:
            df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
    return df

def detect_outliers(df, column_name, threshold=3):
    """Detect outliers using z-score method."""
    if column_name in df.columns:
        z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
        return df[z_scores < threshold]
    return df

def clean_dataframe(df, operations=None):
    """Apply multiple cleaning operations to DataFrame."""
    if operations is None:
        operations = ['remove_duplicates', 'fill_missing']
    
    cleaned_df = df.copy()
    
    if 'remove_duplicates' in operations:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if 'fill_missing' in operations:
        cleaned_df = fill_missing_values(cleaned_df)
    
    return cleaned_df
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original statistics:")
    original_stats = calculate_statistics(df, 'value')
    for key, value in original_stats.items():
        print(f"{key}: {value:.2f}")
    
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print(f"\nCleaned shape: {cleaned_df.shape}")
    print(f"Removed {len(df) - len(cleaned_df)} outliers")
    
    print("\nCleaned statistics:")
    cleaned_stats = calculate_statistics(cleaned_df, 'value')
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    example_usage()