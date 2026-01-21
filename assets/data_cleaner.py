
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def get_cleaning_stats(original_df, cleaned_df):
    """
    Get statistics about the cleaning process.
    
    Parameters:
    original_df (pd.DataFrame): Original DataFrame
    cleaned_df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
    dict: Dictionary containing cleaning statistics
    """
    stats = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data with outliers
    data = {
        'temperature': np.concatenate([np.random.normal(20, 5, 90), [100, -15, 150]]),
        'humidity': np.concatenate([np.random.normal(50, 10, 90), [200, -30]]),
        'pressure': np.random.normal(1013, 5, 93)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics:")
    print(df.describe())
    
    # Clean the data
    cleaned_df = clean_numeric_data(df)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    print(cleaned_df.describe())
    
    # Get cleaning statistics
    stats = get_cleaning_stats(df, cleaned_df)
    print("\nCleaning statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")