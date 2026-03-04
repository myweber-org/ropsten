
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (list or np.array): Input data array
        column (int): Column index to process (for 2D arrays)
    
    Returns:
        np.array: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if data.ndim == 2:
        column_data = data[:, column]
    else:
        column_data = data
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if data.ndim == 2:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return data[mask]
    else:
        return column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (np.array): Input data array
    
    Returns:
        dict: Dictionary containing mean, median, std, min, max
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    return stats

def clean_dataset(data, column_index=0):
    """
    Main function to clean dataset by removing outliers.
    
    Args:
        data (list or np.array): Input dataset
        column_index (int): Column to check for outliers
    
    Returns:
        tuple: (cleaned_data, removed_count, original_stats, cleaned_stats)
    """
    original_shape = np.array(data).shape
    original_stats = calculate_statistics(data)
    
    cleaned_data = remove_outliers_iqr(data, column_index)
    removed_count = len(data) - len(cleaned_data)
    cleaned_stats = calculate_statistics(cleaned_data)
    
    return cleaned_data, removed_count, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = np.random.randn(100, 3) * 10 + 50
    sample_data[0:5, 0] = [200, 250, -100, 300, 150]
    
    cleaned, removed, orig_stats, clean_stats = clean_dataset(sample_data, 0)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned.shape}")
    print(f"Removed outliers: {removed}")
    print(f"Original stats: {orig_stats}")
    print(f"Cleaned stats: {clean_stats}")