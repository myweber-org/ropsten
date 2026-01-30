
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int or str): Column index or name if data is structured.
    
    Returns:
    np.ndarray: Data with outliers removed.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return filtered_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (list or array-like): The dataset.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = [10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 10, 100, 12, 14, 13, 12, 11, 14, 13, 12]
    
    print("Original data:", sample_data)
    print("Original statistics:", calculate_statistics(sample_data))
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    print("Cleaned data:", cleaned_data)
    print("Cleaned statistics:", calculate_statistics(cleaned_data))
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process DataFrame by removing outliers from multiple numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of numeric column names to process
    
    Returns:
        pd.DataFrame: Processed DataFrame with outliers removed
    """
    processed_df = df.copy()
    
    for column in numeric_columns:
        if column in processed_df.columns:
            original_count = len(processed_df)
            processed_df = remove_outliers_iqr(processed_df, column)
            removed_count = original_count - len(processed_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return processed_df