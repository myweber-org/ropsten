import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        data: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def zscore_normalize(data, column):
    """
    Normalize a column using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize a column using min-max normalization.
    
    Args:
        data: pandas DataFrame
        column: Column name to normalize
        feature_range: Desired range of transformed data (default 0-1)
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return pd.Series([feature_range[0]] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    scaled = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return scaled

def detect_missing_patterns(data, threshold=0.3):
    """
    Detect columns with high percentage of missing values.
    
    Args:
        data: pandas DataFrame
        threshold: Missing value threshold (default 0.3 = 30%)
    
    Returns:
        List of columns exceeding the missing value threshold
    """
    missing_ratios = data.isnull().sum() / len(data)
    high_missing_cols = missing_ratios[missing_ratios > threshold].index.tolist()
    
    return high_missing_cols

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_multiplier: IQR multiplier for outlier removal
        normalize_method: 'zscore' or 'minmax' normalization
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            
            if normalize_method == 'zscore':
                cleaned_data[f'{column}_normalized'] = zscore_normalize(cleaned_data, column)
            elif normalize_method == 'minmax':
                cleaned_data[f'{column}_normalized'] = minmax_normalize(cleaned_data, column)
    
    high_missing = detect_missing_patterns(cleaned_data)
    if high_missing:
        print(f"Warning: Columns with >30% missing values: {high_missing}")
    
    return cleaned_data