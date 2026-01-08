
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column].apply(lambda x: 0)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    if columns is None:
        columns = data.columns
    
    cleaned_data = data.copy()
    
    for col in columns:
        if cleaned_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = cleaned_data[col].mean()
            elif strategy == 'median':
                fill_value = cleaned_data[col].median()
            elif strategy == 'mode':
                fill_value = cleaned_data[col].mode()[0]
            elif strategy == 'drop':
                cleaned_data = cleaned_data.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            cleaned_data[col] = cleaned_data[col].fillna(fill_value)
    
    return cleaned_data

def validate_dataframe(data, required_columns=None, dtypes=None):
    """
    Validate dataframe structure and data types
    """
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if dtypes:
        for col, expected_type in dtypes.items():
            if col in data.columns:
                actual_type = data[col].dtype
                if actual_type != expected_type:
                    try:
                        data[col] = data[col].astype(expected_type)
                    except ValueError:
                        raise TypeError(f"Column {col} cannot be cast to {expected_type}")
    
    return True