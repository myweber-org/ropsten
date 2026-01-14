
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean multiple numeric columns in a dataset by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = {
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.normal(60, 15, 100),
        'pressure': np.random.normal(1013, 10, 100)
    }
    
    # Add some outliers
    data['temperature'][0] = 100
    data['humidity'][1] = 200
    data['pressure'][2] = 500
    
    df = pd.DataFrame(data)
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal summary statistics:")
    for col in df.columns:
        stats = calculate_summary_statistics(df, col)
        print(f"\n{col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
    
    cleaned_df = clean_dataset(df, ['temperature', 'humidity', 'pressure'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    for col in cleaned_df.columns:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"\n{col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")