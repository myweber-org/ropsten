
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding the threshold percentage.
    
    Args:
        df: pandas DataFrame
        threshold: float between 0 and 1, default 0.5
    
    Returns:
        Cleaned DataFrame
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    missing_per_row = df.isnull().mean(axis=1)
    return df[missing_per_row <= threshold].reset_index(drop=True)

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        DataFrame with filled values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df_filled[col] = df[col].fillna(median_val)
    
    return df_filled

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def clean_dataset(df, missing_threshold=0.3, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_threshold: threshold for removing rows with missing values
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    print(f"Original shape: {df.shape}")
    
    # Step 1: Remove rows with excessive missing values
    df_clean = remove_missing_rows(df, threshold=missing_threshold)
    print(f"After missing value removal: {df_clean.shape}")
    
    # Step 2: Fill remaining missing values with median
    df_clean = fill_missing_with_median(df_clean)
    
    # Step 3: Remove outliers
    df_clean = remove_outliers_iqr(df_clean, multiplier=outlier_multiplier)
    print(f"After outlier removal: {df_clean.shape}")
    
    # Step 4: Standardize numeric columns
    df_clean = standardize_columns(df_clean)
    
    return df_clean