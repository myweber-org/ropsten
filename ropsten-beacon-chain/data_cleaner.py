
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns.
    """
    clean_data = data.copy()
    for col in columns:
        outliers = detect_outliers_iqr(clean_data, col, threshold)
        clean_data = clean_data[~outliers]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    """
    normalized_data = data.copy()
    for col in columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data

def normalize_zscore(data, columns):
    """
    Apply z-score normalization to specified columns.
    """
    normalized_data = data.copy()
    for col in columns:
        normalized_data[col] = stats.zscore(normalized_data[col])
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    if columns is None:
        columns = data.columns
    
    processed_data = data.copy()
    
    for col in columns:
        if processed_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_data[col].mean()
            elif strategy == 'median':
                fill_value = processed_data[col].median()
            elif strategy == 'mode':
                fill_value = processed_data[col].mode()[0]
            elif strategy == 'drop':
                processed_data = processed_data.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            processed_data[col] = processed_data[col].fillna(fill_value)
    
    return processed_data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, 
                  normalization='minmax', missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    # Handle missing values
    cleaned_data = handle_missing_values(data, strategy=missing_strategy, 
                                         columns=numeric_columns)
    
    # Remove outliers
    cleaned_data = remove_outliers(cleaned_data, numeric_columns, 
                                   threshold=outlier_threshold)
    
    # Apply normalization
    if normalization == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data, numeric_columns)
    elif normalization == 'zscore':
        cleaned_data = normalize_zscore(cleaned_data, numeric_columns)
    
    return cleaned_data

def validate_data(data, numeric_columns):
    """
    Validate cleaned data for common issues.
    """
    validation_report = {}
    
    for col in numeric_columns:
        validation_report[col] = {
            'has_nan': data[col].isnull().any(),
            'has_inf': np.isinf(data[col]).any(),
            'min': data[col].min(),
            'max': data[col].max(),
            'mean': data[col].mean(),
            'std': data[col].std()
        }
    
    return validation_report
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result