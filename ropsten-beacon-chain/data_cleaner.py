
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling null values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default is True.
    column_case (str): Target case for column names ('lower', 'upper', 'title'). Default is 'lower'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for identifying duplicates. Default is None (all columns).
    keep (str): Which duplicates to keep ('first', 'last', False). Default is 'first'.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    df_deduped = df.drop_duplicates(subset=subset, keep=keep)
    df_deduped = df_deduped.reset_index(drop=True)
    
    return df_deduped

def convert_column_types(df, column_type_map):
    """
    Convert data types of specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_type_map (dict): Dictionary mapping column names to target data types.
    
    Returns:
    pd.DataFrame: DataFrame with converted column types.
    """
    df_converted = df.copy()
    
    for column, dtype in column_type_map.items():
        if column in df_converted.columns:
            try:
                df_converted[column] = df_converted[column].astype(dtype)
            except Exception as e:
                print(f"Error converting column {column} to {dtype}: {e}")
    
    return df_converted