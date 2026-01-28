
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric column names to clean
        outlier_method: 'iqr' or 'zscore'
        normalize_method: 'minmax' or 'zscore'
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        # Remove outliers
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        # Normalize data
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_cleaning(df, original_df, numeric_columns):
    """
    Validate the cleaning process by comparing statistics.
    
    Args:
        df: cleaned DataFrame
        original_df: original DataFrame
        numeric_columns: list of numeric column names
    
    Returns:
        Dictionary with validation metrics
    """
    validation_results = {}
    
    for col in numeric_columns:
        if col not in df.columns or col not in original_df.columns:
            continue
            
        validation_results[col] = {
            'original_mean': original_df[col].mean(),
            'cleaned_mean': df[col].mean(),
            'original_std': original_df[col].std(),
            'cleaned_std': df[col].std(),
            'original_range': original_df[col].max() - original_df[col].min(),
            'cleaned_range': df[col].max() - df[col].min(),
            'rows_removed': len(original_df) - len(df)
        }
    
    return validation_results