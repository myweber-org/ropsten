import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    """Normalize specified column to range [0,1]."""
    if column_name in df.columns:
        col = df[column_name]
        df[column_name] = (col - col.min()) / (col.max() - col.min())
    return df

def clean_dataset(file_path, output_path=None):
    """Main cleaning pipeline for CSV datasets."""
    df = pd.read_csv(file_path)
    
    df = remove_duplicates(df)
    df = fill_missing_values(df, strategy='median')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df = normalize_column(df, col)
    
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df