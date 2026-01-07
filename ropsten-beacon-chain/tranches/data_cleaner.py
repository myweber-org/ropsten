
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
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
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val != 0:
                standardized_data[col] = (data[col] - mean_val) / std_val
    
    return standardized_data

def clean_dataset(data, outlier_method='iqr', normalization_method='minmax', 
                  outlier_columns=None, norm_columns=None, outlier_params=None):
    """
    Main function to clean dataset with outlier removal and normalization
    """
    cleaned_data = data.copy()
    
    if outlier_columns is None:
        outlier_columns = data.select_dtypes(include=[np.number]).columns
    
    if outlier_method == 'iqr':
        factor = outlier_params.get('factor', 1.5) if outlier_params else 1.5
        for col in outlier_columns:
            if col in data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col, factor)
    elif outlier_method == 'zscore':
        threshold = outlier_params.get('threshold', 3) if outlier_params else 3
        for col in outlier_columns:
            if col in data.columns:
                cleaned_data = remove_outliers_zscore(cleaned_data, col, threshold)
    
    if normalization_method == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data, norm_columns)
    elif normalization_method == 'zscore':
        cleaned_data = normalize_zscore(cleaned_data, norm_columns)
    
    return cleaned_data

def validate_data(data, check_missing=True, check_duplicates=True):
    """
    Validate data quality
    """
    validation_report = {}
    
    if check_missing:
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        validation_report['missing_values'] = missing_values[missing_values > 0]
        validation_report['missing_percentage'] = missing_percentage[missing_percentage > 0]
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_count
    
    return validation_report