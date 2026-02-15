
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, convert_types=True):
    """
    Clean a pandas DataFrame by removing duplicates and converting data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    convert_types (bool): Whether to convert columns to optimal data types.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if convert_types:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                print(f"Converted column '{col}' to datetime.")
            except (ValueError, TypeError):
                try:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                    print(f"Converted column '{col}' to numeric.")
                except (ValueError, TypeError):
                    pass
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'dtypes': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_result['null_counts'][col] = null_count
        
        validation_result['dtypes'][col] = str(df[col].dtype)
    
    return validation_result

def sample_data_processing():
    """
    Example function demonstrating data cleaning workflow.
    """
    data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'score': ['85', '92', '92', '78', '95', '95'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-04']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['id', 'name', 'score'])
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    sample_data_processing()
import pandas as pd

def clean_dataframe(df, column, threshold, keep_above=True):
    """
    Filters a DataFrame based on a numeric threshold in a specified column.
    Returns a new DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if keep_above:
        filtered_df = df[df[column] > threshold].copy()
    else:
        filtered_df = df[df[column] <= threshold].copy()

    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def remove_duplicates_by_column(df, column):
    """
    Removes duplicate rows based on values in a specified column,
    keeping the first occurrence.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    cleaned_df = df.drop_duplicates(subset=[column], keep='first').copy()
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: multiplier for IQR (default 1.5)
    
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
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    
    valid_indices = data[column].dropna().index[mask]
    return data.loc[valid_indices]

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
    
    col_data = data[column].copy()
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return col_data
    
    return (col_data - min_val) / (max_val - min_val)

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
    
    col_data = data[column].copy()
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return col_data
    
    return (col_data - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric columns)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        if normalize_method == 'minmax':
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f"{column}_normalized"] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
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
    
    summary = pd.DataFrame({
        'mean': numeric_data.mean(),
        'median': numeric_data.median(),
        'std': numeric_data.std(),
        'min': numeric_data.min(),
        'max': numeric_data.max(),
        'q1': numeric_data.quantile(0.25),
        'q3': numeric_data.quantile(0.75),
        'skewness': numeric_data.skew(),
        'kurtosis': numeric_data.kurtosis(),
        'missing_values': numeric_data.isnull().sum(),
        'zero_values': (numeric_data == 0).sum()
    })
    
    return summary.T

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Args:
        data: pandas DataFrame
        threshold: absolute skewness threshold (default: 0.5)
    
    Returns:
        List of skewed column names
    """
    numeric_data = data.select_dtypes(include=[np.number])
    skewness = numeric_data.skew().abs()
    
    skewed_columns = skewness[skewness > threshold].index.tolist()
    return skewed_columns

def apply_log_transform(data, columns):
    """
    Apply log transformation to specified columns.
    
    Args:
        data: pandas DataFrame
        columns: list of column names to transform
    
    Returns:
        DataFrame with log-transformed columns
    """
    transformed_data = data.copy()
    
    for column in columns:
        if column in transformed_data.columns:
            min_val = transformed_data[column].min()
            if min_val <= 0:
                transformed_data[f"{column}_log"] = np.log(transformed_data[column] - min_val + 1)
            else:
                transformed_data[f"{column}_log"] = np.log(transformed_data[column])
    
    return transformed_data