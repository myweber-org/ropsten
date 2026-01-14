
import pandas as pd
import numpy as np
from pathlib import Path

def clean_dataset(input_path, output_path=None):
    """
    Load a CSV dataset, remove duplicate rows, normalize column names,
    and save the cleaned version.
    """
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")
    
    df_cleaned = df.copy()
    
    df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(' ', '_')
    
    df_cleaned = df_cleaned.drop_duplicates()
    
    df_cleaned = df_cleaned.replace(r'^\s*$', np.nan, regex=True)
    
    cleaned_shape = df_cleaned.shape
    print(f"Cleaned dataset shape: {cleaned_shape}")
    print(f"Removed {original_shape[0] - cleaned_shape[0]} duplicate rows.")
    print(f"Removed {original_shape[1] - cleaned_shape[1]} duplicate columns.")
    
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
    
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    
    return df_cleaned

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    clean_dataset(input_file, output_file)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_rows': 0,
        'null_values': df.isnull().sum().sum()
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    validation_results['empty_rows'] = df.isnull().all(axis=1).sum()
    
    return validation_results