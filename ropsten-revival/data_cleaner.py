
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
    
    return filtered_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nDataFrame after removing outliers:")
    print(cleaned_df)
    
    stats = calculate_statistics(df, 'values')
    print("\nStatistics for 'values' column:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    normalized_df = normalize_column(df, 'values', method='minmax')
    print("\nDataFrame after min-max normalization:")
    print(normalized_df)import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data.copy()
    
    normalized = data.copy()
    normalized[column] = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data.copy()
    
    standardized = data.copy()
    standardized[column] = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy not in ['mean', 'median', 'mode', 'drop']:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    processed = data.copy()
    numeric_cols = processed.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        processed = processed.dropna(subset=numeric_cols)
    elif strategy == 'mode':
        for col in numeric_cols:
            mode_val = processed[col].mode()
            if not mode_val.empty:
                processed[col].fillna(mode_val[0], inplace=True)
    else:
        for col in numeric_cols:
            if strategy == 'mean':
                fill_val = processed[col].mean()
            else:  # median
                fill_val = processed[col].median()
            processed[col].fillna(fill_val, inplace=True)
    
    return processed

def clean_dataset(data, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        data: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    cleaned = data.copy()
    
    # Handle missing values
    if 'missing_strategy' in config:
        cleaned = handle_missing_values(cleaned, config['missing_strategy'])
    
    # Remove outliers
    if 'outlier_columns' in config:
        for col in config['outlier_columns']:
            if col in cleaned.columns:
                cleaned = remove_outliers_iqr(cleaned, col)
    
    # Normalize columns
    if 'normalize_columns' in config:
        for col in config['normalize_columns']:
            if col in cleaned.columns:
                cleaned = normalize_minmax(cleaned, col)
    
    # Standardize columns
    if 'standardize_columns' in config:
        for col in config['standardize_columns']:
            if col in cleaned.columns:
                cleaned = standardize_zscore(cleaned, col)
    
    return cleaned.reset_index(drop=True)