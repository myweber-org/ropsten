
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column (int): Column index to calculate outliers for.
    
    Returns:
    numpy.ndarray: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the cleaned data.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if data.size == 0:
        return {"mean": None, "median": None, "std": None}
    
    stats = {
        "mean": np.mean(data, axis=0),
        "median": np.median(data, axis=0),
        "std": np.std(data, axis=0)
    }
    
    return stats

def process_dataset(data_path, column_index):
    """
    Main function to load, clean, and analyze dataset.
    
    Parameters:
    data_path (str): Path to the data file.
    column_index (int): Index of column to clean.
    
    Returns:
    tuple: Cleaned data and statistics.
    """
    try:
        data = np.loadtxt(data_path, delimiter=',')
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {data_path}")
    
    cleaned_data = remove_outliers_iqr(data, column_index)
    stats = calculate_statistics(cleaned_data)
    
    return cleaned_data, stats

if __name__ == "__main__":
    sample_data = np.random.randn(100, 3) * 10 + 50
    cleaned = remove_outliers_iqr(sample_data, 1)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    
    sample_stats = calculate_statistics(cleaned)
    for key, value in sample_stats.items():
        print(f"{key}: {value}")