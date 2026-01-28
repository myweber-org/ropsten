
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask where True indicates outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(df, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    Returns cleaned DataFrame.
    """
    df_clean = df.copy()
    outlier_mask = pd.Series([False] * len(df))
    
    for col in columns:
        if col in df.columns:
            outlier_mask |= detect_outliers_iqr(df, col, threshold)
    
    return df_clean[~outlier_mask].reset_index(drop=True)

def normalize_minmax(data, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    Returns normalized array.
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max == data_min:
        return np.zeros_like(data)
    
    normalized = (data - data_min) / (data_max - data_min)
    
    if feature_range != (0, 1):
        min_val, max_val = feature_range
        normalized = normalized * (max_val - min_val) + min_val
    
    return normalized

def standardize_zscore(data):
    """
    Standardize data using Z-score normalization.
    Returns standardized array.
    """
    return stats.zscore(data, nan_policy='omit')

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean.reset_index(drop=True)

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    Returns tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"