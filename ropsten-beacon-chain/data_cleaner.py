
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count(),
        'missing': data[column].isnull().sum()
    }
    
    return stats

def normalize_column(data, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    
    if method == 'minmax':
        min_val = data_copy[column].min()
        max_val = data_copy[column].max()
        if max_val != min_val:
            data_copy[column] = (data_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = data_copy[column].mean()
        std_val = data_copy[column].std()
        if std_val != 0:
            data_copy[column] = (data_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return data_copy

def handle_missing_values(data, column, strategy='mean'):
    """
    Handle missing values in a DataFrame column.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    
    if strategy == 'mean':
        fill_value = data_copy[column].mean()
    elif strategy == 'median':
        fill_value = data_copy[column].median()
    elif strategy == 'mode':
        fill_value = data_copy[column].mode()[0] if not data_copy[column].mode().empty else np.nan
    elif strategy == 'drop':
        data_copy = data_copy.dropna(subset=[column])
        return data_copy
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    data_copy[column] = data_copy[column].fillna(fill_value)
    return data_copy