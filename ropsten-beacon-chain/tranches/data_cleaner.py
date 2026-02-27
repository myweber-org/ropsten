
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', drop_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                            Options: 'mean', 'median', 'mode', 'drop', 'zero'.
    drop_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype in [np.float64, np.int64]:
            if cleaned_df[column].isnull().any():
                missing_count = cleaned_df[column].isnull().sum()
                print(f"Column '{column}' has {missing_count} missing values.")
                
                if missing_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif missing_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif missing_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                elif missing_strategy == 'zero':
                    fill_value = 0
                elif missing_strategy == 'drop':
                    cleaned_df = cleaned_df.dropna(subset=[column])
                    print(f"Dropped rows with missing values in column '{column}'.")
                    continue
                else:
                    raise ValueError(f"Unknown missing_strategy: {missing_strategy}")
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with {fill_value:.2f}")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', np.nan, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(df, missing_strategy='mean', drop_duplicates=True)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nValidation result: {is_valid}")