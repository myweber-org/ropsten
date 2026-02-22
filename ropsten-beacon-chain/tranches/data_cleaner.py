import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame containing data with potential missing values
        strategy: Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns: List of column names to apply cleaning to (None for all columns)
    
    Returns:
        Cleaned pandas DataFrame
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
            if not df_clean[col].mode().empty:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    
    return df_clean

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        Boolean mask indicating outlier rows
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series(False, index=df.index)
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Series with normalized values
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df[column]
    
    if method == 'minmax':
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max == col_min:
            return pd.Series(0.5, index=df.index)
        return (df[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df[column].mean()
        col_std = df[column].std()
        if col_std == 0:
            return pd.Series(0, index=df.index)
        return (df[column] - col_mean) / col_std
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

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
    
    if df.shape[0] < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_missing_data(df, strategy='mean')
    print("\nCleaned DataFrame (mean imputation):")
    print(cleaned_df)
    
    outliers = detect_outliers_iqr(cleaned_df, 'A')
    print(f"\nOutliers in column A: {outliers.sum()}")
    
    normalized = normalize_column(cleaned_df, 'A', method='minmax')
    print("\nNormalized column A:")
    print(normalized)
    
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")