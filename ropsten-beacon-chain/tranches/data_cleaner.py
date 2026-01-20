
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 14, 13, 12, 11, 10]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_basic_stats(cleaned_df, 'values'))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max - col_min > 0:
                normalized_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def detect_skewed_columns(dataframe, skew_threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Args:
        dataframe: pandas DataFrame
        skew_threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        Dictionary with skewness values for columns exceeding threshold
    """
    skewed_cols = {}
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > skew_threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def clean_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    cleaned_df = dataframe.copy()
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        if strategy == 'drop':
            cleaned_df = cleaned_df.dropna(subset=[col])
        elif strategy == 'mean' and np.issubdtype(cleaned_df[col].dtype, np.number):
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif strategy == 'median' and np.issubdtype(cleaned_df[col].dtype, np.number):
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif strategy == 'mode':
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"