import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to clean, if None clean all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers
    threshold (float): IQR multiplier threshold
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_data(df, columns=None):
    """
    Standardize numerical columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to standardize
    
    Returns:
    pd.DataFrame: Standardized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        mean = df_standardized[col].mean()
        std = df_standardized[col].std()
        if std > 0:
            df_standardized[col] = (df_standardized[col] - mean) / std
    
    return df_standardized

def normalize_data(df, columns=None, range_min=0, range_max=1):
    """
    Normalize numerical columns to specified range.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize
    range_min (float): Minimum value of normalized range
    range_max (float): Maximum value of normalized range
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        col_min = df_normalized[col].min()
        col_max = df_normalized[col].max()
        
        if col_max > col_min:
            df_normalized[col] = ((df_normalized[col] - col_min) / (col_max - col_min)) * (range_max - range_min) + range_min
    
    return df_normalized

def clean_dataset(df, missing_strategy='mean', remove_outliers=True, standardize=True):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values
    remove_outliers (bool): Whether to remove outliers
    standardize (bool): Whether to standardize data
    
    Returns:
    pd.DataFrame: Cleaned and processed DataFrame
    """
    df_clean = clean_missing_values(df, strategy=missing_strategy)
    
    if remove_outliers:
        df_clean = remove_outliers_iqr(df_clean)
    
    if standardize:
        df_clean = standardize_data(df_clean)
    
    return df_clean