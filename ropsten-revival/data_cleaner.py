
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for duplicates
    keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop' to remove rows, 'fill' to fill values
    fill_value: Value to use when strategy is 'fill'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if strategy == 'drop':
        cleaned_df = df.dropna()
        removed_count = len(df) - len(cleaned_df)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing values")
    elif strategy == 'fill':
        if fill_value is None:
            fill_value = df.mean(numeric_only=True)
        cleaned_df = df.fillna(fill_value)
        print(f"Filled missing values with {fill_value}")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df

def normalize_columns(df, columns=None):
    """
    Normalize specified columns to 0-1 range.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    normalized_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max > col_min:
                normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
                print(f"Normalized column '{col}'")
    
    return normalized_df

def clean_data_pipeline(df, remove_dups=True, handle_missing=True, normalize=True):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    remove_dups (bool): Whether to remove duplicates
    handle_missing (bool): Whether to handle missing values
    normalize (bool): Whether to normalize numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_missing:
        cleaned_df = clean_missing_values(cleaned_df, strategy='fill')
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df