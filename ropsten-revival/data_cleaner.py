
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
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

def normalize_column(dataframe, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        method: 'minmax' or 'zscore' normalization
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'minmax':
        col_min = dataframe[column].min()
        col_max = dataframe[column].max()
        
        if col_max == col_min:
            return dataframe[column].apply(lambda x: 0.5)
        
        normalized = (dataframe[column] - col_min) / (col_max - col_min)
        return normalized
    
    elif method == 'zscore':
        col_mean = dataframe[column].mean()
        col_std = dataframe[column].std()
        
        if col_std == 0:
            return dataframe[column].apply(lambda x: 0)
        
        normalized = (dataframe[column] - col_mean) / col_std
        return normalized
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5, normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        dataframe: Input DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_multiplier: Multiplier for IQR outlier detection
        normalize_method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            # Remove outliers
            q1 = cleaned_df[column].quantile(0.25)
            q3 = cleaned_df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_multiplier * iqr
            upper_bound = q3 + outlier_multiplier * iqr
            
            mask = (cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)
            cleaned_df = cleaned_df[mask]
            
            # Normalize
            if normalize_method == 'minmax':
                col_min = cleaned_df[column].min()
                col_max = cleaned_df[column].max()
                if col_max != col_min:
                    cleaned_df[column] = (cleaned_df[column] - col_min) / (col_max - col_min)
            
            elif normalize_method == 'zscore':
                col_mean = cleaned_df[column].mean()
                col_std = cleaned_df[column].std()
                if col_std != 0:
                    cleaned_df[column] = (cleaned_df[column] - col_mean) / col_std
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"