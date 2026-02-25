
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to check for outliers
    
    Returns:
    numpy.ndarray: Cleaned data without outliers
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise ValueError("Column index out of bounds")
    
    q1 = np.percentile(data[:, column], 25)
    q3 = np.percentile(data[:, column], 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the cleaned dataset.
    
    Parameters:
    data (numpy.ndarray): Input data array
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation
    """
    stats = {
        'mean': np.mean(data, axis=0),
        'median': np.median(data, axis=0),
        'std': np.std(data, axis=0)
    }
    return stats

def validate_data(data):
    """
    Validate data for NaN values and infinite values.
    
    Parameters:
    data (numpy.ndarray): Input data array
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if np.any(np.isnan(data)):
        return False
    if np.any(np.isinf(data)):
        return False
    return True

if __name__ == "__main__":
    sample_data = np.random.randn(100, 3)
    sample_data[10, 1] = 100
    sample_data[20, 2] = -50
    
    print("Original data shape:", sample_data.shape)
    
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("Cleaned data shape:", cleaned.shape)
    
    if validate_data(cleaned):
        stats = calculate_statistics(cleaned)
        print("Statistics:", stats)