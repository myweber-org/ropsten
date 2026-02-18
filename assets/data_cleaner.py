
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from specified column
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_column(data, column, method='minmax'):
    """
    Normalize column using specified method
    """
    if method == 'minmax':
        min_val = data[column].min()
        max_val = data[column].max()
        data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = data[column].mean()
        std_val = data[column].std()
        data[column + '_normalized'] = (data[column] - mean_val) / std_val
    
    return data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers(cleaned_data, column, outlier_threshold)
            
            if normalize:
                cleaned_data = normalize_column(cleaned_data, column, method='zscore')
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate statistical summary of dataset
    """
    summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns)
    }
    
    return summary

def validate_dataframe(data):
    """
    Validate dataframe structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    return True