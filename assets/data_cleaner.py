
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (np.ndarray): Input data array
        column (int): Column index to process
    
    Returns:
        np.ndarray: Data with outliers removed
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

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Args:
        data (np.ndarray): Input dataset
        columns_to_clean (list): List of column indices to clean
    
    Returns:
        np.ndarray: Cleaned dataset
    """
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column < cleaned_data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data

def calculate_statistics(data):
    """
    Calculate basic statistics for each column in the dataset.
    
    Args:
        data (np.ndarray): Input dataset
    
    Returns:
        dict: Dictionary containing statistics for each column
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

if __name__ == "__main__":
    sample_data = np.random.randn(100, 3)
    sample_data[0, 0] = 100
    sample_data[1, 1] = -50
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_statistics(sample_data))
    
    cleaned = clean_dataset(sample_data, [0, 1, 2])
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned))