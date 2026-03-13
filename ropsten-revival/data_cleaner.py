
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check for missing values. 
                 If None, checks all columns.
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to fill. If None, fills all numeric columns.
    
    Returns:
        DataFrame with filled values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            df_filled[col] = df[col].fillna(df[col].mean())
    
    return df_filled

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check for outliers.
                 If None, checks all numeric columns.
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame without outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have mean=0 and std=1.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to standardize.
                 If None, standardizes all numeric columns.
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                df_standardized[col] = (df[col] - mean) / std
    
    return df_standardized

def get_data_summary(df):
    """
    Generate summary statistics for DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    return summary
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: Specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if df_copy[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a column in DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        threshold: IQR multiplier threshold
    
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def clean_data_pipeline(df: pd.DataFrame, 
                       remove_dup: bool = True,
                       handle_na: bool = True,
                       na_strategy: str = 'mean',
                       normalize_cols: List[str] = None,
                       norm_method: str = 'minmax') -> pd.DataFrame:
    """
    Complete data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dup: Whether to remove duplicates
        handle_na: Whether to handle missing values
        na_strategy: Strategy for handling missing values
        normalize_cols: Columns to normalize
        norm_method: Normalization method
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dup:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_na:
        cleaned_df = handle_missing_values(cleaned_df, strategy=na_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col, method=norm_method)
    
    return cleaned_dfimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using the IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        else:
            raise ValueError("Outlier method must be 'iqr' or 'zscore'")
    
    for col in numeric_columns:
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
        else:
            raise ValueError("Normalize method must be 'minmax' or 'zscore'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    numeric_cols = ['feature1', 'feature2']
    cleaned_data = clean_dataset(sample_data, numeric_cols, outlier_method='zscore', normalize_method='zscore')
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(cleaned_data.describe())import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a cleaned Series with outliers set to NaN.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series")
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned = data.copy()
    cleaned[(cleaned < lower_bound) | (cleaned > upper_bound)] = np.nan
    return cleaned

def normalize_minmax(data):
    """
    Normalize data using min-max scaling to range [0, 1].
    Handles NaN values by ignoring them in calculation.
    """
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise TypeError("Input must be array-like")
    
    data_array = np.array(data, dtype=float)
    valid_mask = ~np.isnan(data_array)
    
    if not np.any(valid_mask):
        return np.full_like(data_array, np.nan)
    
    valid_data = data_array[valid_mask]
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    
    if data_max == data_min:
        normalized = np.zeros_like(data_array)
    else:
        normalized = (data_array - data_min) / (data_max - data_min)
    
    normalized[~valid_mask] = np.nan
    return normalized

def winsorize_data(data, limits=(0.05, 0.05)):
    """
    Apply winsorization to limit extreme values.
    Uses scipy.stats.mstats.winsorize for efficient processing.
    """
    try:
        from scipy.stats.mstats import winsorize
    except ImportError:
        raise ImportError("scipy is required for winsorization")
    
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise TypeError("Input must be array-like")
    
    data_array = np.ma.array(data, dtype=float)
    winsorized = winsorize(data_array, limits=limits)
    return winsorized.data

def clean_dataframe(df, columns=None, methods=None):
    """
    Apply cleaning methods to specified columns of a DataFrame.
    Supports 'iqr', 'normalize', and 'winsorize' methods.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    if methods is None:
        methods = {col: 'iqr' for col in columns}
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        method = methods.get(col, 'iqr')
        original_data = cleaned_df[col]
        
        if method == 'iqr':
            cleaned_df[col] = remove_outliers_iqr(original_data, col)
        elif method == 'normalize':
            cleaned_df[col] = normalize_minmax(original_data)
        elif method == 'winsorize':
            cleaned_df[col] = winsorize_data(original_data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return cleaned_df

def get_cleaning_summary(original_df, cleaned_df):
    """
    Generate summary statistics comparing original and cleaned data.
    """
    summary = {}
    
    for col in original_df.select_dtypes(include=[np.number]).columns:
        if col not in cleaned_df.columns:
            continue
            
        orig = original_df[col]
        clean = cleaned_df[col]
        
        summary[col] = {
            'original_count': len(orig),
            'cleaned_count': len(clean),
            'outliers_removed': orig.isna().sum() - clean.isna().sum(),
            'original_mean': orig.mean(),
            'cleaned_mean': clean.mean(),
            'original_std': orig.std(),
            'cleaned_std': clean.std()
        }
    
    return pd.DataFrame(summary).T

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 100),
        'B': np.random.exponential(50, 100),
        'C': np.random.uniform(0, 1, 100)
    })
    
    # Add some outliers
    sample_data.loc[10:15, 'A'] = 500
    sample_data.loc[20:25, 'B'] = 1000
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:")
    print(sample_data.describe())
    
    cleaned = clean_dataframe(sample_data, methods={'A': 'iqr', 'B': 'winsorize', 'C': 'normalize'})
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned statistics:")
    print(cleaned.describe())
    
    summary = get_cleaning_summary(sample_data, cleaned)
    print("\nCleaning summary:")
    print(summary)import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df, column_types):
    """
    Convert specified columns to given data types.
    
    Args:
        df: pandas DataFrame
        column_types: dict mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_converted = df.copy()
    for column, dtype in column_types.items():
        if column in df_converted.columns:
            df_converted[column] = df_converted[column].astype(dtype)
    return df_converted

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def normalize_numeric_columns(df, columns):
    """
    Normalize numeric columns to range [0, 1].
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    for column in columns:
        if column in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[column]):
            col_min = df_normalized[column].min()
            col_max = df_normalized[column].max()
            if col_max != col_min:
                df_normalized[column] = (df_normalized[column] - col_min) / (col_max - col_min)
    return df_normalized

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: pandas DataFrame
        operations: list of tuples (operation_name, kwargs)
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    operation_map = {
        'remove_duplicates': remove_duplicates,
        'convert_types': convert_column_types,
        'handle_missing': handle_missing_values,
        'normalize': normalize_numeric_columns
    }
    
    for op_name, kwargs in operations:
        if op_name in operation_map:
            cleaned_df = operation_map[op_name](cleaned_df, **kwargs)
    
    return cleaned_df

def validate_dataframe(df, rules):
    """
    Validate DataFrame against specified rules.
    
    Args:
        df: pandas DataFrame
        rules: dict with validation rules
    
    Returns:
        dict with validation results
    """
    results = {}
    
    if 'required_columns' in rules:
        missing = set(rules['required_columns']) - set(df.columns)
        results['missing_columns'] = list(missing)
    
    if 'numeric_ranges' in rules:
        range_violations = {}
        for column, (min_val, max_val) in rules['numeric_ranges'].items():
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                violations = df[(df[column] < min_val) | (df[column] > max_val)]
                if not violations.empty:
                    range_violations[column] = len(violations)
        results['range_violations'] = range_violations
    
    return resultsimport pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_types: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_converted = df.copy()
    for column, dtype in column_types.items():
        if column in df_converted.columns:
            try:
                df_converted[column] = df_converted[column].astype(dtype)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert column '{column}' to {dtype}")
    
    return df_converted

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'drop',
                          fill_value: Union[int, float, str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    df_cleaned = df.copy()
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif strategy == 'fill' and fill_value is not None:
        df_cleaned = df_cleaned.fillna(fill_value)
    
    return df_cleaned

def clean_dataframe(df: pd.DataFrame,
                    deduplicate: bool = True,
                    type_conversions: dict = None,
                    missing_strategy: str = 'drop',
                    fill_value: Union[int, float, str] = None) -> pd.DataFrame:
    """
    Main function to clean DataFrame with multiple operations.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary for column type conversions
        missing_strategy: Strategy for handling missing values
        fill_value: Value to fill missing values with
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, missing_strategy, fill_value)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        True if DataFrame is valid, False otherwise
    """
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True