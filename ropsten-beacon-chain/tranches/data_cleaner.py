
import pandas as pd
import numpy as np
from typing import Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[list] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: Specific columns to fill, None for all columns
    
    Returns:
        DataFrame with filled missing values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df_filled[col] = df[col].fillna(0)
        else:
            df_filled[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
    
    return df_filled

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a numeric column to range [0, 1].
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    df_normalized = df.copy()
    
    if column in df.columns and df[column].dtype in ['int64', 'float64']:
        col_min = df[column].min()
        col_max = df[column].max()
        
        if col_max != col_min:
            df_normalized[column] = (df[column] - col_min) / (col_max - col_min)
        else:
            df_normalized[column] = 0.5
    
    return df_normalized

def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        threshold: IQR multiplier threshold
    
    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
        return pd.Series([False] * len(df), index=df.index)
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def clean_dataframe(df: pd.DataFrame, 
                   remove_dup: bool = True,
                   fill_na: bool = True,
                   fill_strategy: str = 'mean',
                   normalize_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dup: Whether to remove duplicates
        fill_na: Whether to fill missing values
        fill_strategy: Strategy for filling missing values
        normalize_cols: Columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dup:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy=fill_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
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

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to normalize (default: all numeric columns)
    
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
            
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max != col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0
    
    return normalized_df

def standardize_zscore(dataframe, columns=None):
    """
    Standardize specified columns using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to standardize (default: all numeric columns)
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    standardized_df = dataframe.copy()
    
    for col in columns:
        if col not in standardized_df.columns:
            continue
            
        col_mean = standardized_df[col].mean()
        col_std = standardized_df[col].std()
        
        if col_std > 0:
            standardized_df[col] = (standardized_df[col] - col_mean) / col_std
        else:
            standardized_df[col] = 0
    
    return standardized_df

def clean_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        dataframe: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: List of columns to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    cleaned_df = dataframe.copy()
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        if cleaned_df[col].isnull().any():
            if strategy == 'drop':
                cleaned_df = cleaned_df.dropna(subset=[col])
            elif strategy == 'mean':
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif strategy == 'mode':
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
    
    return cleaned_df

def process_data_pipeline(dataframe, config):
    """
    Execute a data cleaning pipeline based on configuration.
    
    Args:
        dataframe: Input DataFrame
        config: Dictionary with processing configuration
    
    Returns:
        Processed DataFrame
    """
    df = dataframe.copy()
    
    if 'missing_values' in config:
        df = clean_missing_values(df, **config['missing_values'])
    
    if 'outliers' in config:
        for outlier_config in config['outliers']:
            df = remove_outliers_iqr(df, **outlier_config)
    
    if 'normalization' in config:
        norm_type = config['normalization'].get('type', 'minmax')
        
        if norm_type == 'minmax':
            df = normalize_minmax(df, **config['normalization'])
        elif norm_type == 'zscore':
            df = standardize_zscore(df, **config['normalization'])
    
    return df