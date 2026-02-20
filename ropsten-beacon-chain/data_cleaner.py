
import pandas as pd

def clean_dataframe(df, text_columns=None):
    """
    Clean a pandas DataFrame by removing rows with null values
    and standardizing text columns to lowercase.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize specified text columns to lowercase
    if text_columns:
        for col in text_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype(str).str.lower()
    
    return df_cleaned

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def validate_numeric_range(df, column, min_val=None, max_val=None):
    """
    Validate that values in a numeric column fall within specified range.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if min_val is not None:
        df = df[df[column] >= min_val]
    
    if max_val is not None:
        df = df[df[column] <= max_val]
    
    return df