
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

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method=None, missing_strategy='mean'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, col)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, col)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    return cleaned_df

def validate_data(df, required_columns, numeric_ranges=None):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if numeric_ranges:
        for col, (min_val, max_val) in numeric_ranges.items():
            if col in df.columns:
                invalid_values = df[(df[col] < min_val) | (df[col] > max_val)]
                if not invalid_values.empty:
                    raise ValueError(f"Column {col} contains values outside range [{min_val}, {max_val}]")
    
    return Trueimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is True.
        fill_value: Value to use for filling missing values. Default is 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', fill_value=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'mode', 'constant', 'drop'.
        fill_value: Value to use when strategy is 'constant'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_shape = df.shape
    
    if strategy == 'drop':
        df_cleaned = df.dropna()
    elif strategy in ['mean', 'median', 'mode']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            fill_values = df[numeric_cols].mean()
        elif strategy == 'median':
            fill_values = df[numeric_cols].median()
        elif strategy == 'mode':
            fill_values = df[numeric_cols].mode().iloc[0]
        
        df_cleaned = df.copy()
        df_cleaned[numeric_cols] = df[numeric_cols].fillna(fill_values)
        
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if not non_numeric_cols.empty:
            df_cleaned[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("fill_value must be provided when using 'constant' strategy")
        df_cleaned = df.fillna(fill_value)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    cleaned_shape = df_cleaned.shape
    missing_count = df.isna().sum().sum()
    
    print(f"Original data shape: {original_shape}")
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Missing values handled: {missing_count}")
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        output_path (str): Path for the output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "sample_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, strategy='mean')
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")
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
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max != col_min:
                result_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                result_df[col] = 0
    
    return result_df

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Args:
        dataframe: pandas DataFrame
        threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        Dictionary with column names and their skewness values
    """
    skewed_cols = {}
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def clean_dataset(dataframe, outlier_columns=None, normalize=True, skew_threshold=0.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame to clean
        outlier_columns: list of columns for outlier removal (default: all numeric)
        normalize: whether to normalize numeric columns
        skew_threshold: skewness detection threshold
    
    Returns:
        Cleaned DataFrame and cleaning report
    """
    original_shape = dataframe.shape
    
    if outlier_columns is None:
        outlier_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in outlier_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df)
    
    skewed_cols = detect_skewed_columns(cleaned_df, skew_threshold)
    
    report = {
        'original_samples': original_shape[0],
        'cleaned_samples': cleaned_df.shape[0],
        'removed_samples': original_shape[0] - cleaned_df.shape[0],
        'skewed_columns': skewed_cols,
        'normalization_applied': normalize
    }
    
    return cleaned_df, report