
import pandas as pd

def clean_dataset(df, missing_strategy='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    remove_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if df_clean.isnull().sum().any():
        print("Handling missing values...")
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if col in numeric_cols:
                    if missing_strategy == 'mean':
                        fill_value = df_clean[col].mean()
                    elif missing_strategy == 'median':
                        fill_value = df_clean[col].median()
                    elif missing_strategy == 'mode':
                        fill_value = df_clean[col].mode()[0]
                    elif missing_strategy == 'drop':
                        df_clean = df_clean.dropna(subset=[col])
                        continue
                    else:
                        raise ValueError(f"Unknown strategy: {missing_strategy}")
                    
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"Filled missing values in '{col}' with {missing_strategy}: {fill_value:.2f}")
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
                    print(f"Filled missing values in '{col}' with 'Unknown'")
    
    print(f"Cleaning complete. Final shape: {df_clean.shape}")
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 4],
        'B': [5, None, 7, 8, 8],
        'C': ['x', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, missing_strategy='mean', remove_duplicates=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid = validate_data(cleaned_df, required_columns=['A', 'B', 'C'], min_rows=1)
    print(f"\nData validation passed: {is_valid}")