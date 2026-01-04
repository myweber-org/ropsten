
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Cleans a pandas DataFrame by removing duplicate rows and standardizing column names.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def handle_missing_values(df, strategy='mean'):
    """
    Handles missing values in numeric columns using specified strategy.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df_filled = df.copy()
        for col in numeric_cols:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    elif strategy == 'median':
        df_filled = df.copy()
        for col in numeric_cols:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    else:
        df_filled = df.copy()
    
    return df_filled

def validate_dataframe(df):
    """
    Validates that the DataFrame meets basic requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True