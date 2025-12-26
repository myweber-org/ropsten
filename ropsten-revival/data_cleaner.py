
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and normalizing text columns.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if normalize_text:
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    return df_clean

def validate_data(df, required_columns=None, unique_constraints=None):
    """
    Validate data integrity by checking required columns and unique constraints.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if unique_constraints:
        for constraint in unique_constraints:
            if not isinstance(constraint, list):
                constraint = [constraint]
            
            duplicate_mask = df.duplicated(subset=constraint, keep=False)
            if duplicate_mask.any():
                duplicate_count = duplicate_mask.sum()
                raise ValueError(
                    f"Duplicate values found in constraint {constraint}: {duplicate_count} rows"
                )
    
    return True

def sample_data(df, sample_size=1000, random_state=42):
    """
    Create a random sample from the dataset for testing purposes.
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)