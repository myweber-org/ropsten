
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric columns by removing outliers and filling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            # Fill missing values with median
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
        'C': [100, 200, 300, 400, 500, 600, 700, 800, 900, 10000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics for column 'A':")
    print(calculate_summary_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, columns=['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standard'):
    cleaned_df = df.copy()
    
    if outlier_removal:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalization == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization == 'standard':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

def get_summary_statistics(df, numeric_columns):
    summary = {}
    for col in numeric_columns:
        if col in df.columns:
            summary[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count()
            }
    return pd.DataFrame(summary).T
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default True.
    column_case (str): Target case for column names ('lower', 'upper', 'title'). Default 'lower'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    else:
        # Fill numeric columns with median, object columns with mode
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Standardize column names
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    # Remove leading/trailing whitespace from column names
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # Replace spaces with underscores in column names
    cleaned_df.columns = cleaned_df.columns.str.replace(' ', '_')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'Name': ['Alice', 'Bob', None, 'David'],
#         'Age': [25, None, 35, 28],
#         'City': ['NYC', 'LA', 'Chicago', None]
#     }
#     df = pd.DataFrame(sample_data)
#     cleaned = clean_dataset(df, drop_na=False)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataframe(cleaned, required_columns=['name', 'age'])
#     print(f"\nValidation: {is_valid}, Message: {message}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def zscore_normalize(dataframe, column):
    """
    Normalize a column using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column]
    
    normalized = (dataframe[column] - mean_val) / std_val
    return normalized

def minmax_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize a column using min-max normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val == min_val:
        return dataframe[column]
    
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    
    target_min, target_max = feature_range
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
        elif strategy == 'mean':
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strategy == 'median':
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strategy == 'mode':
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_copy

def create_cleaning_pipeline(dataframe, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        dataframe: pandas DataFrame to clean
        config: dictionary with cleaning operations configuration
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = dataframe.copy()
    
    if 'remove_outliers' in config:
        for col in config['remove_outliers'].get('columns', []):
            df_clean = remove_outliers_iqr(
                df_clean, 
                col, 
                config['remove_outliers'].get('multiplier', 1.5)
            )
    
    if 'normalize' in config:
        for col, method in config['normalize'].items():
            if method == 'zscore':
                df_clean[f"{col}_normalized"] = zscore_normalize(df_clean, col)
            elif method == 'minmax':
                df_clean[f"{col}_normalized"] = minmax_normalize(df_clean, col)
    
    if 'handle_missing' in config:
        df_clean = handle_missing_values(
            df_clean,
            strategy=config['handle_missing'].get('strategy', 'mean'),
            columns=config['handle_missing'].get('columns')
        )
    
    return df_clean
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result