
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.array: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    col_data = data[:, column].astype(float)
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    cleaned_data = remove_outliers_iqr(data, column)
    col_data = cleaned_data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'sample_count': len(col_data)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = np.array([
        [1, 150.5],
        [2, 162.3],
        [3, 145.8],
        [4, 210.1],
        [5, 138.9],
        [6, 155.2],
        [7, 300.7],
        [8, 148.6],
        [9, 152.4],
        [10, 165.0]
    ])
    
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("Cleaned data:")
    print(cleaned)
    
    stats = calculate_statistics(sample_data, 1)
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")