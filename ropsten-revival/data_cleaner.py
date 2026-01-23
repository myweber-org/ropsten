
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Index of column to process
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if data.size == 0:
        return data
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for each column in the data.
    
    Parameters:
    data (numpy.ndarray): Input data array
    
    Returns:
    dict: Dictionary containing mean, median, and std for each column
    """
    if data.size == 0:
        return {}
    
    stats = {}
    for i in range(data.shape[1]):
        col_data = data[:, i]
        stats[f'column_{i}'] = {
            'mean': np.mean(col_data),
            'median': np.median(col_data),
            'std': np.std(col_data),
            'min': np.min(col_data),
            'max': np.max(col_data)
        }
    
    return stats

def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    numpy.ndarray: Normalized data
    """
    if data.size == 0:
        return data
    
    normalized = np.zeros_like(data, dtype=float)
    
    for i in range(data.shape[1]):
        col_data = data[:, i]
        
        if method == 'minmax':
            col_min = np.min(col_data)
            col_max = np.max(col_data)
            if col_max - col_min > 0:
                normalized[:, i] = (col_data - col_min) / (col_max - col_min)
        
        elif method == 'zscore':
            col_mean = np.mean(col_data)
            col_std = np.std(col_data)
            if col_std > 0:
                normalized[:, i] = (col_data - col_mean) / col_std
    
    return normalized

def process_data_pipeline(data, outlier_column=0, normalize_method='minmax'):
    """
    Complete data processing pipeline: remove outliers and normalize.
    
    Parameters:
    data (numpy.ndarray): Input data array
    outlier_column (int): Column index for outlier removal
    normalize_method (str): Normalization method
    
    Returns:
    tuple: (cleaned_data, statistics)
    """
    cleaned_data = remove_outliers_iqr(data, outlier_column)
    normalized_data = normalize_data(cleaned_data, normalize_method)
    stats = calculate_statistics(normalized_data)
    
    return normalized_data, stats