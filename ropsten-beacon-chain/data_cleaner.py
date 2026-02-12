import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    Returns a boolean mask for outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(df, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    Returns cleaned DataFrame.
    """
    df_clean = df.copy()
    for col in columns:
        outlier_mask = detect_outliers_iqr(df_clean, col, threshold)
        df_clean = df_clean[~outlier_mask]
    return df_clean.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    df_norm = data.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def standardize_zscore(data, columns):
    """
    Apply z-score standardization to specified columns.
    Returns DataFrame with standardized columns.
    """
    df_std = data.copy()
    for col in columns:
        mean_val = df_std[col].mean()
        std_val = df_std[col].std()
        if std_val > 0:
            df_std[col] = (df_std[col] - mean_val) / std_val
    return df_std

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_clean.dropna(subset=columns)
    
    for col in columns:
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif strategy == 'median':
                fill_value = df_clean[col].median()
            elif strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, 
                  normalize=False, standardize=False, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    """
    # Handle missing values
    df_clean = handle_missing_values(df, strategy=missing_strategy, 
                                     columns=numeric_columns)
    
    # Remove outliers
    df_clean = remove_outliers(df_clean, numeric_columns, outlier_threshold)
    
    # Apply normalization or standardization
    if normalize:
        df_clean = normalize_minmax(df_clean, numeric_columns)
    elif standardize:
        df_clean = standardize_zscore(df_clean, numeric_columns)
    
    return df_clean

def validate_data(df, numeric_columns):
    """
    Validate data quality after cleaning.
    Returns dictionary with validation metrics.
    """
    validation = {
        'total_rows': len(df),
        'missing_values': df[numeric_columns].isnull().sum().to_dict(),
        'data_types': df[numeric_columns].dtypes.astype(str).to_dict(),
        'basic_stats': {}
    }
    
    for col in numeric_columns:
        validation['basic_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    return validation