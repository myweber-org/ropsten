
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to process
        factor: multiplier for IQR (default 1.5)
    
    Returns:
        Cleaned DataFrame with outliers removed
    """
    df_clean = dataframe.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(dataframe, columns):
    """
    Normalize data using Min-Max scaling to [0, 1] range.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    df_norm = dataframe.copy()
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        
        if max_val != min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    
    return df_norm

def standardize_zscore(dataframe, columns):
    """
    Standardize data using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    df_std = dataframe.copy()
    
    for col in columns:
        if col not in df_std.columns:
            continue
            
        mean_val = df_std[col].mean()
        std_val = df_std[col].std()
        
        if std_val > 0:
            df_std[col] = (df_std[col] - mean_val) / std_val
        else:
            df_std[col] = 0
    
    return df_std

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    df_processed = dataframe.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        df_processed = df_processed.dropna(subset=columns)
    else:
        for col in columns:
            if col not in df_processed.columns:
                continue
                
            if strategy == 'mean':
                fill_value = df_processed[col].mean()
            elif strategy == 'median':
                fill_value = df_processed[col].median()
            elif strategy == 'mode':
                fill_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 0
            else:
                fill_value = 0
            
            df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed.reset_index(drop=True)

def clean_dataset(dataframe, numeric_columns=None, outlier_factor=1.5, 
                  normalize=False, standardize=False, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_factor: IQR factor for outlier removal
        normalize: whether to apply min-max normalization
        standardize: whether to apply z-score standardization
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned and processed DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = dataframe.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy, columns=numeric_columns)
    df_clean = remove_outliers_iqr(df_clean, columns=numeric_columns, factor=outlier_factor)
    
    if normalize:
        df_clean = normalize_minmax(df_clean, columns=numeric_columns)
    elif standardize:
        df_clean = standardize_zscore(df_clean, columns=numeric_columns)
    
    return df_clean

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
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