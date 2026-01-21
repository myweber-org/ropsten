import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process, None for all numeric columns
    factor (float): Multiplier for IQR
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize, None for all numeric columns
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_norm = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        
        if col_max - col_min == 0:
            df_norm[col] = min_val
        else:
            df_norm[col] = min_val + (df_norm[col] - col_min) * (max_val - min_val) / (col_max - col_min)
    
    return df_norm

def zscore_normalize(df, columns=None, threshold=3):
    """
    Normalize data using Z-score and optionally cap extreme values.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize
    threshold (float): Z-score threshold for capping
    
    Returns:
    pd.DataFrame: Dataframe with Z-score normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_z = df.copy()
    
    for col in columns:
        mean_val = df_z[col].mean()
        std_val = df_z[col].std()
        
        if std_val == 0:
            df_z[col] = 0
        else:
            z_scores = (df_z[col] - mean_val) / std_val
            df_z[col] = np.clip(z_scores, -threshold, threshold)
    
    return df_z

def clean_dataset(df, outlier_method='iqr', normalize_method=None, **kwargs):
    """
    Main function to clean dataset with optional outlier removal and normalization.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    outlier_method (str): 'iqr' or None for outlier removal
    normalize_method (str): 'minmax', 'zscore', or None for normalization
    **kwargs: Additional parameters for specific methods
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    if outlier_method == 'iqr':
        factor = kwargs.get('iqr_factor', 1.5)
        columns = kwargs.get('outlier_columns', None)
        df_clean = remove_outliers_iqr(df_clean, columns=columns, factor=factor)
    
    if normalize_method == 'minmax':
        columns = kwargs.get('norm_columns', None)
        feature_range = kwargs.get('feature_range', (0, 1))
        df_clean = normalize_minmax(df_clean, columns=columns, feature_range=feature_range)
    
    elif normalize_method == 'zscore':
        columns = kwargs.get('norm_columns', None)
        threshold = kwargs.get('zscore_threshold', 3)
        df_clean = zscore_normalize(df_clean, columns=columns, threshold=threshold)
    
    return df_clean

def validate_data(df, check_missing=True, check_duplicates=True, check_inf=True):
    """
    Validate data quality and return summary statistics.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    check_missing (bool): Check for missing values
    check_duplicates (bool): Check for duplicate rows
    check_inf (bool): Check for infinite values
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    if check_missing:
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        validation_results['total_missing'] = missing_counts.sum()
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    if check_inf:
        numeric_cols = df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_cols).sum()
        validation_results['infinite_values'] = inf_counts[inf_counts > 0].to_dict()
    
    return validation_results