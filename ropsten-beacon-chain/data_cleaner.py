
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            
            if max_val != min_val:
                normalized_df[col] = (dataframe[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and dataframe[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(dataframe[col]):
                processed_df[col] = dataframe[col].fillna(dataframe[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(dataframe[col]):
                processed_df[col] = dataframe[col].fillna(dataframe[col].median())
            elif strategy == 'mode':
                processed_df[col] = dataframe[col].fillna(dataframe[col].mode()[0])
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
    
    return processed_df

def calculate_statistics(dataframe, columns=None):
    """
    Calculate basic statistics for numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    stats_dict = {}
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            stats_dict[col] = {
                'mean': dataframe[col].mean(),
                'median': dataframe[col].median(),
                'std': dataframe[col].std(),
                'min': dataframe[col].min(),
                'max': dataframe[col].max(),
                'skewness': dataframe[col].skew(),
                'kurtosis': dataframe[col].kurtosis()
            }
    
    return stats_dict

def clean_dataset(dataframe, config=None):
    """
    Comprehensive data cleaning pipeline
    """
    if config is None:
        config = {
            'outlier_removal': True,
            'normalization': True,
            'missing_values': 'mean'
        }
    
    cleaned_df = dataframe.copy()
    
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if config.get('missing_values'):
        cleaned_df = handle_missing_values(
            cleaned_df, 
            strategy=config['missing_values']
        )
    
    if config.get('outlier_removal'):
        for col in numeric_cols:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if config.get('normalization'):
        cleaned_df = normalize_minmax(cleaned_df, numeric_cols)
    
    return cleaned_df