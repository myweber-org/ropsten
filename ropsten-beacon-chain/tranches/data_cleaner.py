import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop' to remove rows, 
                       'ffill' to forward fill, 'bfill' to backward fill.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'ffill':
        cleaned_df = cleaned_df.ffill()
    elif fill_method == 'bfill':
        cleaned_df = cleaned_df.bfill()
    else:
        raise ValueError("fill_method must be 'drop', 'ffill', or 'bfill'")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_count': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, None, 4, 4],
#         'B': [5, None, 7, 8, 8],
#         'C': [9, 10, 11, 12, 12]
#     }
#     df = pd.DataFrame(sample_data)
#     
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df, remove_duplicates=True, fill_method='ffill')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     validation = validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
#     print("\nValidation Results:")
#     for key, value in validation.items():
#         print(f"{key}: {value}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to clean
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
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

def clean_dataset(df, numeric_columns):
    """
    Clean a dataset by removing outliers from multiple numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of numeric column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[95:99, 'value'] = [500, 600, 700, 800, 900]
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics:")
    print(calculate_summary_statistics(df, 'value'))
    
    cleaned_df = clean_dataset(df, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    print(calculate_summary_statistics(cleaned_df, 'value'))import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Args:
        data: pandas Series containing the data
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        Series with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return series[(series >= lower_bound) & (series <= upper_bound)]

def zscore_normalize(data, column):
    """
    Normalize data using z-score normalization.
    
    Args:
        data: pandas Series containing the data
        column: column name to normalize
    
    Returns:
        Normalized Series
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return series - mean
    
    return (series - mean) / std

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas Series containing the data
        column: column name to normalize
        feature_range: desired range of transformed data (default 0-1)
    
    Returns:
        Normalized Series
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    min_val = series.min()
    max_val = series.max()
    
    if min_val == max_val:
        return series * 0 + feature_range[0]
    
    scaled = (series - min_val) / (max_val - min_val)
    return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method=None):
    """
    Clean a dataset by handling outliers and optionally normalizing numeric columns.
    
    Args:
        df: pandas DataFrame to clean
        numeric_columns: list of numeric column names to process (default: all numeric)
        outlier_method: 'iqr' or None for outlier removal
        normalize_method: 'zscore', 'minmax', or None for normalization
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            mask = ~cleaned_df[col].isna()
            cleaned_series = remove_outliers_iqr(cleaned_df[mask], col)
            cleaned_df.loc[mask, col] = cleaned_series.reindex(cleaned_df[mask].index)
        
        if normalize_method == 'zscore':
            cleaned_df[col] = zscore_normalize(cleaned_df, col)
        elif normalize_method == 'minmax':
            cleaned_df[col] = minmax_normalize(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan=True, max_nan_ratio=0.5):
    """
    Validate dataset structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether NaN values are allowed
        max_nan_ratio: maximum allowed ratio of NaN values per column
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing required columns: {missing}"
    
    if not allow_nan:
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            return False, f"NaN values found in columns: {nan_columns}"
    else:
        for col in df.columns:
            nan_ratio = df[col].isna().mean()
            if nan_ratio > max_nan_ratio:
                return False, f"Column '{col}' has too many NaN values ({nan_ratio:.1%})"
    
    return True, "Data validation passed"