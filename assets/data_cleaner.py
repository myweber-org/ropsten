
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_summary_statistics(data):
    """
    Generate summary statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return pd.DataFrame()
    
    summary = numeric_data.describe().T
    summary['skewness'] = numeric_data.skew()
    summary['kurtosis'] = numeric_data.kurtosis()
    summary['missing'] = numeric_data.isnull().sum()
    summary['missing_pct'] = (summary['missing'] / len(data)) * 100
    
    return summaryimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Parameters:
    data (pd.Series): Input data series
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.Series: Data with outliers removed
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input must be a pandas Series")
    
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def normalize_minmax(data):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Parameters:
    data (np.ndarray or pd.Series): Input data
    
    Returns:
    np.ndarray: Normalized data
    """
    if len(data) == 0:
        return np.array([])
    
    data_array = np.array(data)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    if max_val == min_val:
        return np.zeros_like(data_array)
    
    return (data_array - min_val) / (max_val - min_val)

def standardize_zscore(data):
    """
    Standardize data using z-score normalization.
    
    Parameters:
    data (np.ndarray or pd.Series): Input data
    
    Returns:
    np.ndarray: Standardized data with mean=0, std=1
    """
    if len(data) == 0:
        return np.array([])
    
    data_array = np.array(data)
    mean_val = np.mean(data_array)
    std_val = np.std(data_array)
    
    if std_val == 0:
        return np.zeros_like(data_array)
    
    return (data_array - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize=False):
    """
    Clean a dataset by handling outliers and optionally normalizing.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to process
    outlier_method (str): Method for outlier removal ('iqr' or None)
    normalize (bool): Whether to normalize the data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            mask = ~cleaned_df[col].isna()
            cleaned_data = remove_outliers_iqr(cleaned_df.loc[mask, col], col)
            cleaned_df.loc[mask, col] = cleaned_data
            
        if normalize:
            mask = ~cleaned_df[col].isna()
            cleaned_df.loc[mask, col] = normalize_minmax(cleaned_df.loc[mask, col])
    
    return cleaned_df

def calculate_statistics(data):
    """
    Calculate descriptive statistics for data.
    
    Parameters:
    data (pd.Series or np.ndarray): Input data
    
    Returns:
    dict: Dictionary containing statistics
    """
    if len(data) == 0:
        return {}
    
    data_array = np.array(data)
    
    stats_dict = {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array),
        'q1': np.percentile(data_array, 25),
        'q3': np.percentile(data_array, 75),
        'skewness': stats.skew(data_array) if len(data_array) > 2 else 0,
        'kurtosis': stats.kurtosis(data_array) if len(data_array) > 3 else 0
    }
    
    return stats_dict

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_statistics(sample_data['A']))
    
    cleaned_data = clean_dataset(sample_data, outlier_method='iqr', normalize=True)
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_statistics(cleaned_data['A'].dropna()))