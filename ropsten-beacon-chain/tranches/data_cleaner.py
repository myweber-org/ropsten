
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
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
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, max, and IQR.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'count': df[column].count(),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75),
        'iqr': df[column].quantile(0.75) - df[column].quantile(0.25)
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    print(f"Original statistics:")
    original_stats = calculate_summary_statistics(df, 'value')
    for key, val in original_stats.items():
        print(f"  {key}: {val:.2f}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned statistics:")
    cleaned_stats = calculate_summary_statistics(cleaned_df, 'value')
    for key, val in cleaned_stats.items():
        print(f"  {key}: {val:.2f}")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'constant').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if strategy == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif strategy == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif strategy == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                elif strategy == 'constant':
                    cleaned_df[column].fillna(0, inplace=True)
    
    return cleaned_df

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns using IQR or Z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to process. If None, process all numeric columns.
    method (str): Method for outlier detection ('iqr' or 'zscore').
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    filtered_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (filtered_df[column] >= lower_bound) & (filtered_df[column] <= upper_bound)
                filtered_df = filtered_df[mask]
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                filtered_df = filtered_df[z_scores <= threshold]
    
    return filtered_df

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    method (str): Normalization method ('minmax' or 'standard').
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    normalized_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if method == 'minmax':
                min_val = df[column].min()
                max_val = df[column].max()
                if max_val != min_val:
                    normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
            elif method == 'standard':
                mean_val = df[column].mean()
                std_val = df[column].std()
                if std_val != 0:
                    normalized_df[column] = (df[column] - mean_val) / std_val
    
    return normalized_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7, 8, 9, 100],
        'B': [10, 20, 20, 40, 50, 60, 70, 80, 90, 1000],
        'C': ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df, strategy='median')
    print(cleaned)
    print("\nDataFrame without outliers:")
    no_outliers = remove_outliers(cleaned, threshold=2.0)
    print(no_outliers)
    print("\nNormalized DataFrame:")
    normalized = normalize_data(no_outliers, method='minmax')
    print(normalized)