import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fillna_strategy='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fillna_strategy (str): Strategy for filling missing values.
                               Options: 'mean', 'median', 'mode', 'zero', 'drop'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fillna_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fillna_strategy in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fillna_strategy == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fillna_strategy}.")
    elif fillna_strategy == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
        print("Filled missing values with mode.")
    elif fillna_strategy == 'zero':
        cleaned_df = cleaned_df.fillna(0)
        print("Filled missing values with zero.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        dict: Validation results.
    """
    validation_result = {
        'is_valid': True,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing_cols
    
    return validation_result

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None, 6],
        'B': [10, None, 30, 40, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Result:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataset(df, fillna_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)