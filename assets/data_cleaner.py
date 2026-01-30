
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int or str): Column index or name if data is structured.
    
    Returns:
    np.ndarray: Data with outliers removed.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return filtered_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (list or array-like): The dataset.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = [10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 10, 100, 12, 14, 13, 12, 11, 14, 13, 12]
    
    print("Original data:", sample_data)
    print("Original statistics:", calculate_statistics(sample_data))
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    print("Cleaned data:", cleaned_data)
    print("Cleaned statistics:", calculate_statistics(cleaned_data))