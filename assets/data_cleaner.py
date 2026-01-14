
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): Index of the column to clean.
    
    Returns:
    numpy.ndarray: Dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds.")
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def example_usage():
    """
    Example usage of the remove_outliers_iqr function.
    """
    np.random.seed(42)
    sample_data = np.random.randn(100, 3)
    sample_data[:, 1] = sample_data[:, 1] * 10 + 50
    
    print("Original data shape:", sample_data.shape)
    cleaned = remove_outliers_iqr(sample_data, column=1)
    print("Cleaned data shape:", cleaned.shape)
    print("Number of outliers removed:", sample_data.shape[0] - cleaned.shape[0])

if __name__ == "__main__":
    example_usage()