
import pandas as pd
import re

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, convert to lowercase).
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_names (list): List of column names to normalize
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize specified string columns
    for col in column_names:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using regex pattern.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with valid email flag column added
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].astype(str).str.match(email_pattern)
    
    return df

def remove_outliers_iqr(df, column_name):
    """
    Remove outliers from a numeric column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of numeric column to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    return df_filtered.reset_index(drop=True)