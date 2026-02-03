import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def validate_dataframe(df):
    """
    Basic validation of DataFrame structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    if df.shape[0] == 0:
        return False
    return True

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate dtype.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.lower()
    df.drop_duplicates(subset=[column_name], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def remove_special_characters(df, column_name):
    """
    Remove special characters from a column, keeping only alphanumeric and spaces.
    """
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    return df

def normalize_column(df, column_name):
    """
    Apply all cleaning functions to a column.
    """
    df = clean_dataframe(df, column_name)
    df = remove_special_characters(df, column_name)
    return df

if __name__ == "__main__":
    sample_data = {'Name': ['  Alice  ', 'Bob', 'alice', 'Charlie!', '  david  ']}
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = normalize_column(df, 'Name')
    print("\nCleaned DataFrame:")
    print(cleaned_df)