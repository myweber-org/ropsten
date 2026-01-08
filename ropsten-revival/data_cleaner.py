
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to clean.
    
    Returns:
    np.array: Data with outliers removed.
    """
    data = np.array(data)
    col_data = data[:, column].astype(float)
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column.
    
    Returns:
    dict: Dictionary containing mean, median, and std.
    """
    data = np.array(data)
    col_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = [
        [1, 150.5],
        [2, 160.2],
        [3, 155.8],
        [4, 165.3],
        [5, 170.1],
        [6, 200.5],
        [7, 50.2]
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("\nCleaned data:")
    for row in cleaned:
        print(row)
    
    stats = calculate_statistics(cleaned, 1)
    print("\nStatistics for cleaned column:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")