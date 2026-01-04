
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            col_min = data[col].min()
            col_max = data[col].max()
            
            if col_max != col_min:
                normalized_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def normalize_zscore(data, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            col_mean = data[col].mean()
            col_std = data[col].std()
            
            if col_std != 0:
                standardized_data[col] = (data[col] - col_mean) / col_std
            else:
                standardized_data[col] = 0
    
    return standardized_data

def clean_dataset(data, outlier_method='iqr', outlier_columns=None, 
                  normalize_method='minmax', normalize_columns=None,
                  outlier_params=None, normalize_params=None):
    """
    Comprehensive data cleaning pipeline
    """
    if outlier_params is None:
        outlier_params = {}
    if normalize_params is None:
        normalize_params = {}
    
    cleaned_data = data.copy()
    stats_report = {}
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_data.columns:
                if outlier_method == 'iqr':
                    cleaned_data, removed = remove_outliers_iqr(cleaned_data, col, **outlier_params)
                elif outlier_method == 'zscore':
                    cleaned_data, removed = remove_outliers_zscore(cleaned_data, col, **outlier_params)
                else:
                    raise ValueError(f"Unknown outlier method: {outlier_method}")
                
                stats_report[f'outliers_removed_{col}'] = removed
    
    if normalize_columns is not None:
        if normalize_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, normalize_columns, **normalize_params)
        elif normalize_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, normalize_columns, **normalize_params)
        else:
            raise ValueError(f"Unknown normalization method: {normalize_method}")
    
    stats_report['original_rows'] = len(data)
    stats_report['cleaned_rows'] = len(cleaned_data)
    stats_report['rows_removed'] = len(data) - len(cleaned_data)
    
    return cleaned_data, stats_report

def validate_data(data, required_columns=None, numeric_columns=None, 
                  allow_nan=True, max_nan_ratio=0.1):
    """
    Validate data quality
    """
    validation_report = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_report['missing_columns'] = missing_columns
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                non_numeric.append(col)
        validation_report['non_numeric_columns'] = non_numeric
    
    if not allow_nan:
        nan_counts = data.isnull().sum()
        high_nan_columns = nan_counts[nan_counts > 0].index.tolist()
        validation_report['columns_with_nan'] = high_nan_columns
    else:
        nan_ratios = data.isnull().mean()
        problematic_columns = nan_ratios[nan_ratios > max_nan_ratio].index.tolist()
        validation_report['high_nan_columns'] = problematic_columns
    
    validation_report['total_rows'] = len(data)
    validation_report['total_columns'] = len(data.columns)
    validation_report['total_nan'] = data.isnull().sum().sum()
    
    return validation_report