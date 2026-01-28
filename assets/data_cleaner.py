
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding threshold percentage.
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Maximum allowed missing percentage per row (0-1)
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    missing_per_row = df.isnull().mean(axis=1)
    return df[missing_per_row <= threshold].reset_index(drop=True)

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Specific columns to fill, None for all numeric columns
    
    Returns:
        pd.DataFrame: Dataframe with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            median_val = df[col].median()
            df_filled[col] = df[col].fillna(median_val)
    
    return df_filled

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Specific columns to check, None for all numeric columns
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: Dataframe without outliers
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            mask = mask & col_mask
    
    return df_clean[mask].reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Specific columns to standardize
    
    Returns:
        pd.DataFrame: Dataframe with standardized columns
    """
    df_std = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df_std[col] = (df[col] - mean_val) / std_val
    
    return df_std

def clean_dataset(df, missing_threshold=0.3, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe
        missing_threshold (float): Threshold for missing value removal
        outlier_multiplier (float): Multiplier for outlier detection
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print(f"Initial shape: {df.shape}")
    
    # Step 1: Remove rows with excessive missing values
    df_clean = remove_missing_rows(df, threshold=missing_threshold)
    print(f"After missing value removal: {df_clean.shape}")
    
    # Step 2: Fill remaining missing values
    df_clean = fill_missing_with_median(df_clean)
    
    # Step 3: Remove outliers
    df_clean = remove_outliers_iqr(df_clean, multiplier=outlier_multiplier)
    print(f"After outlier removal: {df_clean.shape}")
    
    # Step 4: Standardize numeric columns
    df_clean = standardize_columns(df_clean)
    
    return df_clean