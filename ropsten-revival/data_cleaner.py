import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of columns to fill (None for all numeric columns)
    
    Returns:
        DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df_filled[col] = df[col].fillna(0)
    
    return df_filled

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize (None for all numeric columns)
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def detect_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Detect outliers in DataFrame columns.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check (None for all numeric columns)
        method: 'iqr' or 'zscore'
        threshold: multiplier for IQR or cutoff for z-score
    
    Returns:
        Dictionary with outlier counts per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = {}
    
    for col in columns:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_count = (z_scores > threshold).sum()
            
            outliers[col] = outlier_count
    
    return outliers

def clean_dataframe(df, remove_dups=True, fill_na=True, normalize=False, 
                    outlier_threshold=None, subset_cols=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        remove_dups: whether to remove duplicates
        fill_na: whether to fill missing values
        normalize: whether to normalize numeric columns
        outlier_threshold: if provided, remove outliers using IQR method
        subset_cols: specific columns to process (None for all)
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if subset_cols:
        cleaned_df = cleaned_df[subset_cols]
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy='mean')
    
    if outlier_threshold is not None:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & 
                                   (cleaned_df[col] <= upper_bound)]
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df, method='minmax')
    
    return cleaned_dfimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fillna_method='drop'):
    """
    Clean a pandas DataFrame by handling null values and removing duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fillna_method (str): Method to handle null values - 'drop', 'fill_mean', 'fill_median', or 'fill_zero'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle null values
    if fillna_method == 'drop':
        df_clean = df_clean.dropna()
    elif fillna_method == 'fill_mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif fillna_method == 'fill_median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif fillna_method == 'fill_zero':
        df_clean = df_clean.fillna(0)
    
    # Remove duplicates if requested
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'A': [1, 2, None, 4, 2],
#         'B': [5, None, 7, 8, 5],
#         'C': [9, 10, 11, 12, 9]
#     }
#     df = pd.DataFrame(data)
#     
#     # Clean the data
#     cleaned_df = clean_dataset(df, drop_duplicates=True, fillna_method='fill_mean')
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     # Validate the cleaned data
#     is_valid, message = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
#     print(f"\nValidation: {is_valid} - {message}")