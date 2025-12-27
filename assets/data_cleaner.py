
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a column using IQR method.
    
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
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        columns: list of columns to normalize (default: all numeric columns)
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_data = data.copy()
    
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            col_min = data[col].min()
            col_max = data[col].max()
            
            if col_max != col_min:
                normalized_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def standardize_zscore(data, columns=None):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        columns: list of columns to standardize (default: all numeric columns)
    
    Returns:
        Standardized DataFrame
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    standardized_data = data.copy()
    
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            col_mean = data[col].mean()
            col_std = data[col].std()
            
            if col_std != 0:
                standardized_data[col] = (data[col] - col_mean) / col_std
            else:
                standardized_data[col] = 0
    
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process (default: all columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.columns.tolist()
    
    processed_data = data.copy()
    
    for col in columns:
        if col not in processed_data.columns:
            continue
            
        if strategy == 'drop':
            processed_data = processed_data.dropna(subset=[col])
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(processed_data[col]):
            processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(processed_data[col]):
            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        elif strategy == 'mode':
            mode_value = processed_data[col].mode()
            if not mode_value.empty:
                processed_data[col] = processed_data[col].fillna(mode_value.iloc[0])
        else:
            processed_data[col] = processed_data[col].fillna(0)
    
    return processed_data

def clean_dataset(data, outlier_columns=None, normalize=True, handle_missing=True):
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        outlier_columns: columns to remove outliers from
        normalize: whether to normalize numeric columns
        handle_missing: whether to handle missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if handle_missing:
        cleaned_data = handle_missing_values(cleaned_data, strategy='mean')
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    if normalize:
        cleaned_data = normalize_minmax(cleaned_data)
    
    return cleaned_data