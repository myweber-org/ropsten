
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data: numpy array or list-like data structure
        column: index or key of the column to process
        
    Returns:
        Cleaned data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        data_array = np.array(data)
    else:
        data_array = data.copy()
    
    column_data = data_array[:, column] if data_array.ndim > 1 else data_array
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if data_array.ndim > 1:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        cleaned_data = data_array[mask]
    else:
        cleaned_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]
    
    return cleaned_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data: numpy array or list-like data
        
    Returns:
        Dictionary containing mean, median, std, min, and max
    """
    if not isinstance(data, np.ndarray):
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
    # Example usage
    sample_data = np.random.randn(100, 3)
    sample_data[10, 1] = 100  # Add an outlier
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics for column 1:", calculate_statistics(sample_data[:, 1]))
    
    cleaned_data = remove_outliers_iqr(sample_data, 1)
    
    print("Cleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics for column 1:", calculate_statistics(cleaned_data[:, 1]))