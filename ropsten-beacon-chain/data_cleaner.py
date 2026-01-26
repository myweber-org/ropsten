
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (DataFrame): The pandas DataFrame containing the data.
    column (str): The column name to process.
    
    Returns:
    DataFrame: DataFrame with outliers removed from the specified column.
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
    
    Parameters:
    data (DataFrame): The pandas DataFrame.
    column (str): The column name.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    sample_data = pd.DataFrame({
        'values': [10, 12, 12, 13, 14, 15, 15, 16, 17, 100]
    })
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("Original data:")
    print(sample_data)
    print("\nCleaned data:")
    print(cleaned_data)
    print("\nBasic statistics for cleaned data:")
    print(calculate_basic_stats(cleaned_data, 'values'))