
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Args:
        data: pandas DataFrame containing the data.
        column: Name of the column to process.
    
    Returns:
        DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Args:
        data: pandas DataFrame.
        column: Name of the column.
    
    Returns:
        Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column])
    }
    
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': [10, 12, 12, 13, 14, 15, 16, 17, 18, 100]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("\nCleaned data:")
    print(cleaned_data)
    
    stats = calculate_basic_stats(cleaned_data, 'values')
    print("\nStatistics after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")