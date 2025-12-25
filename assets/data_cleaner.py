
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to check for outliers
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    q1 = np.percentile(data[:, column], 25)
    q3 = np.percentile(data[:, column], 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a data column.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    column_data = data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'q1': np.percentile(column_data, 25),
        'q3': np.percentile(column_data, 75)
    }
    
    return stats

def validate_data(data):
    """
    Validate data for cleaning operations.
    
    Parameters:
    data: Input data to validate
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if data is None:
        return False
    
    if not isinstance(data, np.ndarray):
        return False
    
    if data.size == 0:
        return False
    
    if np.any(np.isnan(data)):
        return False
    
    return True

def clean_dataset(data, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (numpy.ndarray): Input data array
    columns_to_clean (list): List of column indices to clean
    
    Returns:
    numpy.ndarray: Cleaned data array
    """
    if not validate_data(data):
        raise ValueError("Invalid input data")
    
    if columns_to_clean is None:
        columns_to_clean = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column < data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data