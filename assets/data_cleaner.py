
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame
        strategy: Method for imputation ('mean', 'median', 'mode', 'drop')
        columns: List of columns to process, None processes all columns
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() == 0:
            continue
        
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'mode':
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col].fillna(mode_val[0], inplace=True)
    
    return df_clean

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a specific column.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold value for outlier detection
    
    Returns:
        Boolean mask indicating outliers
    """
    if column not in df.columns:
        return pd.Series([False] * len(df))
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            z_scores = np.abs((df[column] - mean_val) / std_val)
            outliers = z_scores > threshold
        else:
            outliers = pd.Series([False] * len(df))
    
    return outliers

def normalize_column(df, column, method='minmax'):
    """
    Normalize values in a column.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'standard')
    
    Returns:
        Series with normalized values
    """
    if column not in df.columns:
        return pd.Series()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            normalized = (df[column] - min_val) / (max_val - min_val)
        else:
            normalized = df[column]
    elif method == 'standard':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            normalized = (df[column] - mean_val) / std_val
        else:
            normalized = df[column]
    
    return normalized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def handle_missing_values(df, strategy='mean'):
    handled_df = df.copy()
    numeric_cols = handled_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if handled_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = handled_df[col].mean()
            elif strategy == 'median':
                fill_value = handled_df[col].median()
            elif strategy == 'mode':
                fill_value = handled_df[col].mode()[0]
            else:
                fill_value = 0
            handled_df[col].fillna(fill_value, inplace=True)
    
    return handled_df

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization=True, missing_strategy='mean'):
    df_cleaned = df.copy()
    
    if outlier_removal:
        df_cleaned = remove_outliers_iqr(df_cleaned, numeric_columns)
    
    df_cleaned = handle_missing_values(df_cleaned, strategy=missing_strategy)
    
    if normalization:
        df_cleaned = normalize_minmax(df_cleaned, numeric_columns)
    
    return df_cleaned