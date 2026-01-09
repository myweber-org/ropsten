
import re
import pandas as pd
from typing import List, Optional

def clean_text_column(series: pd.Series, 
                     lowercase: bool = True, 
                     remove_punct: bool = False,
                     strip_whitespace: bool = True) -> pd.Series:
    """
    Clean text data in a pandas Series by applying specified transformations.
    """
    cleaned = series.astype(str)
    
    if strip_whitespace:
        cleaned = cleaned.str.strip()
    
    if lowercase:
        cleaned = cleaned.str.lower()
    
    if remove_punct:
        cleaned = cleaned.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    
    return cleaned

def remove_duplicate_rows(df: pd.DataFrame, 
                         subset: Optional[List[str]] = None, 
                         keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame with configurable options.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_missing_values(df: pd.DataFrame, 
                              na_values: List[str] = None) -> pd.DataFrame:
    """
    Replace various missing value representations with standard NaN.
    """
    if na_values is None:
        na_values = ['', 'NA', 'N/A', 'null', 'NULL', 'NaN', 'nan', 'None']
    
    return df.replace(na_values, pd.NA)

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def create_clean_pipeline(df: pd.DataFrame,
                         text_columns: List[str] = None,
                         **clean_options) -> pd.DataFrame:
    """
    Apply a series of cleaning operations to a DataFrame.
    """
    df_clean = df.copy()
    
    df_clean = standardize_missing_values(df_clean)
    
    if text_columns:
        for col in text_columns:
            df_clean[col] = clean_text_column(df_clean[col], **clean_options)
    
    df_clean = remove_duplicate_rows(df_clean)
    
    validate_dataframe(df_clean)
    
    return df_clean