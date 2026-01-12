
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data.reset_index(drop=True)

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    
    return stats

def process_numerical_column(data, column):
    """
    Complete processing pipeline for a numerical column.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    
    Returns:
    tuple: (cleaned_data, original_stats, cleaned_stats)
    """
    original_stats = calculate_basic_stats(data, column)
    cleaned_data = remove_outliers_iqr(data, column)
    cleaned_stats = calculate_basic_stats(cleaned_data, column)
    
    return cleaned_data, original_stats, cleaned_stats