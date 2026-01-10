import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column].apply(lambda x: 0)
    return df[column].apply(lambda x: (x - mean_val) / std_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[col + '_standardized'] = standardize_zscore(cleaned_df, col)
    return cleaned_df

def summary_statistics(df):
    stats_dict = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': stats.skew(df[col].dropna()),
            'kurtosis': stats.kurtosis(df[col].dropna())
        }
    return pd.DataFrame(stats_dict).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    
    print("Original dataset shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("Cleaned dataset shape:", cleaned.shape)
    
    stats_df = summary_statistics(cleaned.select_dtypes(include=[np.number]))
    print("\nSummary statistics:")
    print(stats_df)import numpy as np
import pandas as pd
from scipy import stats

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
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization == 'standard':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns, min_rows=10):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    return True
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values. Default is True.
        column_case (str): Desired case for column names ('lower', 'upper', or 'title'). Default is 'lower'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Columns to consider for identifying duplicates.
        keep (str): Which duplicates to keep ('first', 'last', or False). Default is 'first'.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

def convert_data_types(df, column_types):
    """
    Convert data types of specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (dict): Dictionary mapping column names to desired data types.
    
    Returns:
        pd.DataFrame: DataFrame with converted data types.
    """
    df_converted = df.copy()
    
    for column, dtype in column_types.items():
        if column in df_converted.columns:
            try:
                df_converted[column] = df_converted[column].astype(dtype)
            except ValueError as e:
                print(f"Warning: Could not convert column '{column}' to {dtype}: {e}")
    
    return df_converted

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, 30, None, 40, 35],
        'Score': [85.5, 92.0, 78.5, 88.0, 91.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = clean_dataset(df, drop_na=True, column_case='lower')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    no_duplicates_df = remove_duplicates(cleaned_df)
    print("DataFrame after removing duplicates:")
    print(no_duplicates_df)
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

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'temperature': np.random.normal(25, 5, 100)
    }
    
    # Add some outliers
    data['value'][10] = 500
    data['value'][20] = -200
    data['temperature'][30] = 100
    
    df = pd.DataFrame(data)
    print(f"Original dataset shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['value', 'temperature'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

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
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (default: all numeric columns)
        outlier_factor: IQR factor for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in data.columns:
            # Remove outliers
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_factor * iqr
            upper_bound = q3 + outlier_factor * iqr
            
            mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
            cleaned_data = cleaned_data[mask]
    
    return cleaned_data.reset_index(drop=True)

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate data structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan and data.isnull().any().any():
        nan_columns = data.columns[data.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_columns}"
    
    return True, "Data validation passed"