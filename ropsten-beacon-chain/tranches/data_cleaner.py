
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    If columns specified, only check those columns.
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            df_filled[col] = df_filled[col].fillna(mean_val)
    return df_filled

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    Returns boolean Series where True indicates outlier.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def cap_outliers(df, column, method='iqr', threshold=1.5):
    """
    Cap outliers to specified bounds.
    Methods: 'iqr' or 'percentile'
    """
    df_capped = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
    elif method == 'percentile':
        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
    else:
        raise ValueError("Method must be 'iqr' or 'percentile'")
    
    df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

def normalize_column(df, column, method='minmax'):
    """
    Normalize column values.
    Methods: 'minmax' or 'zscore'
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_normalized

def clean_dataset(df, missing_strategy='remove', outlier_strategy='cap'):
    """
    Apply comprehensive cleaning pipeline to DataFrame.
    """
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        df_clean = remove_missing_rows(df_clean)
    elif missing_strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean = fill_missing_with_mean(df_clean, numeric_cols)
    
    # Handle outliers for numeric columns
    if outlier_strategy == 'cap':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean = cap_outliers(df_clean, col)
    
    return df_clean