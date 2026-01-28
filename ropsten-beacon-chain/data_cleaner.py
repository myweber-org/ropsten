import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int or str): Column index or name if data is structured.
    
    Returns:
    cleaned_data: Data with outliers removed.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data >= lower_bound) & (data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the cleaned data.
    
    Parameters:
    data (array-like): The cleaned dataset.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)
    
    return {
        'mean': mean_val,
        'median': median_val,
        'standard_deviation': std_val
    }

if __name__ == "__main__":
    sample_data = [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 10, 9, 15, 12, 13, 100]
    cleaned = remove_outliers_iqr(sample_data, 0)
    stats = calculate_statistics(cleaned)
    
    print("Original data:", sample_data)
    print("Cleaned data:", cleaned)
    print("Statistics:", stats)