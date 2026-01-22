
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
    
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a DataFrame column using IQR method.
    
    Args:
        df: pandas DataFrame
        column: Column name to analyze
    
    Returns:
        Boolean mask where True indicates outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def calculate_statistics(df, column):
    """
    Calculate descriptive statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: Column name to analyze
    
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    return stats

def main():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 30, 5)
        ])
    }
    
    df = pd.DataFrame(data)
    print("Original data shape:", df.shape)
    print("Original statistics:", calculate_statistics(df, 'values'))
    
    outliers = detect_outliers_iqr(df, 'values')
    print(f"Number of outliers detected: {outliers.sum()}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("Cleaned data shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned_df, 'values'))

if __name__ == "__main__":
    main()