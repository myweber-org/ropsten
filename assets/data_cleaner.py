import pandas as pd

def clean_dataset(df, columns_to_check=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for nulls. 
            If None, checks all columns.
        remove_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns
    
    for col in columns_to_check:
        if col in cleaned_df.columns:
            cleaned_df = cleaned_df.dropna(subset=[col])
    
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd
from scipy import stats

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
    filtered_indices = np.where(z_scores < threshold)[0]
    
    # Map indices back to original DataFrame
    valid_indices = data[column].dropna().index[filtered_indices]
    filtered_data = data.loc[valid_indices]
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
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    for column in numeric_columns:
        if column in cleaned_data.columns:
            if outlier_method == 'iqr':
                cleaned_data = remove_outliers_iqr(cleaned_data, column)
            elif outlier_method == 'zscore':
                cleaned_data = remove_outliers_zscore(cleaned_data, column)
            else:
                raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    # Normalize data
    for column in numeric_columns:
        if column in cleaned_data.columns:
            if normalize_method == 'minmax':
                cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
            elif normalize_method == 'zscore':
                cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
            else:
                raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=True, max_nan_ratio=0.1):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether NaN values are allowed
        max_nan_ratio: maximum allowed ratio of NaN values per column
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan:
        nan_columns = data.columns[data.isna().any()].tolist()
        if nan_columns:
            return False, f"NaN values found in columns: {nan_columns}"
    else:
        for column in data.columns:
            nan_ratio = data[column].isna().mean()
            if nan_ratio > max_nan_ratio:
                return False, f"Column '{column}' has too many NaN values: {nan_ratio:.1%}"
    
    return True, "Data validation passed"
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): 'iqr' for interquartile range or 'zscore' for standard deviation.
    threshold (float): Threshold value for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        mask = z_scores <= threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to normalize.
    method (str): 'minmax' or 'standard' normalization.
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    data = df_copy[column]
    
    if method == 'minmax':
        min_val = data.min()
        max_val = data.max()
        if max_val != min_val:
            df_copy[column] = (data - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        mean = data.mean()
        std = data.std()
        if std != 0:
            df_copy[column] = (data - mean) / std
    
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")
    
    return df_copy

def validate_dataframe(df, required_columns=None, check_types=True):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    check_types (bool): Whether to check for numeric columns.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    if df.empty:
        validation_results['warnings'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    if check_types:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            validation_results['warnings'].append('No numeric columns found')
    
    return validation_resultsimport pandas as pd
import numpy as np
import re

def clean_csv_data(input_file, output_file):
    """
    Clean CSV data by handling missing values, removing duplicates,
    standardizing text, and converting data types.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Clean text columns
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip() if pd.notnull(x) else x)
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaned successfully. Output saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error cleaning data: {str(e)}")
        return False

def validate_data(df):
    """
    Validate data quality after cleaning.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    return validation_results

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    success = clean_csv_data(input_csv, output_csv)
    
    if success:
        cleaned_df = pd.read_csv(output_csv)
        validation = validate_data(cleaned_df)
        print("Data Validation Results:")
        for key, value in validation.items():
            print(f"{key}: {value}")