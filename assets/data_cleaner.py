
import numpy as np
import pandas as pd

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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data with outliers
    data = {
        'temperature': np.concatenate([
            np.random.normal(20, 2, 90),
            np.array([40, 45, 50, -10, -5])
        ]),
        'humidity': np.concatenate([
            np.random.normal(60, 5, 90),
            np.array([90, 95, 100, 10, 5])
        ]),
        'pressure': np.random.normal(1013, 10, 95)
    }
    
    df = pd.DataFrame(data)
    
    print("Original data shape:", df.shape)
    print("\nOriginal statistics:")
    for col in df.columns:
        stats = calculate_basic_stats(df, col)
        print(f"{col}: {stats}")
    
    # Clean the data
    cleaned_df = clean_numeric_data(df)
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    for col in cleaned_df.columns:
        stats = calculate_basic_stats(cleaned_df, col)
        print(f"{col}: {stats}")