
import pandas as pd
import re

def clean_string_column(series):
    """
    Normalize string column: lowercase, strip whitespace, remove extra spaces.
    """
    if series.dtype == object:
        series = series.astype(str)
        series = series.str.lower()
        series = series.str.strip()
        series = series.apply(lambda x: re.sub(r'\s+', ' ', x))
    return series

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def clean_dataframe(df, string_columns=None):
    """
    Apply cleaning functions to DataFrame.
    """
    df_clean = df.copy()
    
    if string_columns is None:
        string_columns = df_clean.select_dtypes(include=['object']).columns
    
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = clean_string_column(df_clean[col])
    
    df_clean = remove_duplicates(df_clean)
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")