
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
    
    return filtered_df

def zscore_normalize(dataframe, columns=None):
    """
    Normalize specified columns using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        if normalized_df[col].dtype in [np.float64, np.int64]:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            else:
                normalized_df[col] = 0
    
    return normalized_df

def minmax_normalize(dataframe, columns=None, feature_range=(0, 1)):
    """
    Normalize specified columns using min-max normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    output_min, output_max = feature_range
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        if normalized_df[col].dtype in [np.float64, np.int64]:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            
            if max_val > min_val:
                normalized_df[col] = ((normalized_df[col] - min_val) / 
                                     (max_val - min_val)) * (output_max - output_min) + output_min
            else:
                normalized_df[col] = output_min
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'constant', 'drop')
        columns: list of columns to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'mean' and processed_df[col].dtype in [np.float64, np.int64]:
                fill_value = processed_df[col].mean()
            elif strategy == 'median' and processed_df[col].dtype in [np.float64, np.int64]:
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 0
            elif strategy == 'constant':
                fill_value = 0
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def clean_dataset(dataframe, config=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    if config is None:
        config = {
            'outlier_removal': True,
            'normalization': 'zscore',
            'missing_values': 'mean'
        }
    
    cleaned_df = dataframe.copy()
    
    if config.get('missing_values'):
        cleaned_df = handle_missing_values(cleaned_df, strategy=config['missing_values'])
    
    if config.get('outlier_removal'):
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if config.get('normalization'):
        if config['normalization'] == 'zscore':
            cleaned_df = zscore_normalize(cleaned_df)
        elif config['normalization'] == 'minmax':
            cleaned_df = minmax_normalize(cleaned_df)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        tuple of (is_valid, error_message)
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