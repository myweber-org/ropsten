
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = df_clean.columns
    
    missing_counts = {}
    for column in columns_to_check:
        if column in df_clean.columns:
            missing_count = df_clean[column].isnull().sum()
            missing_counts[column] = missing_count
            
            if missing_count > 0:
                if fill_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[column]):
                    fill_value = df_clean[column].mean()
                    df_clean[column].fillna(fill_value, inplace=True)
                elif fill_strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[column]):
                    fill_value = df_clean[column].median()
                    df_clean[column].fillna(fill_value, inplace=True)
                elif fill_strategy == 'mode':
                    fill_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else None
                    df_clean[column].fillna(fill_value, inplace=True)
                elif fill_strategy == 'ffill':
                    df_clean[column].fillna(method='ffill', inplace=True)
                elif fill_strategy == 'bfill':
                    df_clean[column].fillna(method='bfill', inplace=True)
                else:
                    df_clean[column].fillna(0, inplace=True)
    
    # Log cleaning results
    print(f"Removed {removed_duplicates} duplicate rows")
    print(f"Missing values handled using '{fill_strategy}' strategy:")
    for column, count in missing_counts.items():
        print(f"  {column}: {count} missing values filled")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    data = {
        'id': [1, 2, 3, 1, 5, 6, 7, 8],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve', 'Frank', None, 'Helen'],
        'age': [25, 30, 35, 25, 28, None, 40, 32],
        'score': [85.5, 92.0, 78.5, 85.5, None, 88.0, 95.5, 91.0]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, fill_strategy='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
        print("\nData validation passed!")
    except Exception as e:
        print(f"\nData validation failed: {e}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): If True, remove duplicate rows.
        fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
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
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df