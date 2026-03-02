import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return dataframe[(dataframe[column] >= lower_bound) & 
                     (dataframe[column] <= upper_bound)].copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result = dataframe.copy()
    
    for col in columns:
        if col in result.columns and np.issubdtype(result[col].dtype, np.number):
            col_min = result[col].min()
            col_max = result[col].max()
            
            if col_max != col_min:
                result[col] = (result[col] - col_min) / (col_max - col_min)
            else:
                result[col] = 0
    
    return result

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns.tolist()
    
    result = dataframe.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if result[col].isnull().any():
            if strategy == 'drop':
                result = result.dropna(subset=[col])
            elif strategy == 'mean' and np.issubdtype(result[col].dtype, np.number):
                result[col] = result[col].fillna(result[col].mean())
            elif strategy == 'median' and np.issubdtype(result[col].dtype, np.number):
                result[col] = result[col].fillna(result[col].median())
            elif strategy == 'mode':
                result[col] = result[col].fillna(result[col].mode()[0])
    
    return result

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5, 
                  normalize=True, missing_strategy='mean'):
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_multiplier: multiplier for IQR outlier detection
        normalize: whether to apply min-max normalization
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
    
    if normalize and numeric_columns:
        cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    
    return cleaned_df.reset_index(drop=True)