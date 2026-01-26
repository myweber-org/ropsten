
import pandas as pd

def clean_dataset(df, missing_strategy='drop', duplicate_strategy='drop_first'):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                            Options: 'drop', 'fill_mean', 'fill_median', 'fill_mode'.
    duplicate_strategy (str): Strategy for handling duplicates.
                              Options: 'drop_first', 'drop_last', 'keep_none'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill_mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'fill_median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'fill_mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    # Handle duplicates
    if duplicate_strategy == 'drop_first':
        cleaned_df = cleaned_df.drop_duplicates(keep='first')
    elif duplicate_strategy == 'drop_last':
        cleaned_df = cleaned_df.drop_duplicates(keep='last')
    elif duplicate_strategy == 'keep_none':
        cleaned_df = cleaned_df.drop_duplicates(keep=False)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, None, 4, 5, 5],
        'B': [10, 20, 30, None, 50, 50],
        'C': ['x', 'y', 'z', 'x', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='fill_mean', duplicate_strategy='drop_first')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'], min_rows=1)
    print(f"\nValidation: {is_valid}, Message: {message}")