import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
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

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {len(df) - len(df_clean)} duplicate rows.")
    
    if fill_missing:
        missing_before = df_clean.isnull().sum().sum()
        df_clean = df_clean.fillna(fill_value)
        missing_after = df_clean.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values with {fill_value}.")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, np.nan, 20, 30, np.nan],
        'category': ['A', 'B', 'B', 'C', 'A']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    df_clean = clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0)
    
    print("\nCleaned DataFrame:")
    print(df_clean)
    
    # Validate the cleaned data
    try:
        validate_dataframe(df_clean, required_columns=['id', 'value', 'category'])
        print("\nData validation passed.")
    except Exception as e:
        print(f"\nData validation failed: {e}")

if __name__ == "__main__":
    main()