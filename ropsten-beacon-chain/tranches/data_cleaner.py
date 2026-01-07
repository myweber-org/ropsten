def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_order(input_list, key=None):
    if key is None:
        key = lambda x: x
    seen = set()
    result = []
    for item in input_list:
        identifier = key(item)
        if identifier not in seen:
            seen.add(identifier)
            result.append(item)
    return resultimport pandas as pd

def clean_dataframe(df, duplicate_columns=None, missing_strategy='drop'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        duplicate_columns: list of column names to check for duplicates, 
                          if None uses all columns
        missing_strategy: 'drop' to remove rows with missing values,
                         'fill' to fill with column mean (numeric) or mode (categorical)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if duplicate_columns is None:
        duplicate_columns = cleaned_df.columns.tolist()
    
    cleaned_df = cleaned_df.drop_duplicates(subset=duplicate_columns)
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill':
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            else:
                # For categorical/object columns, use mode
                if not cleaned_df[column].mode().empty:
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataframe(df, duplicate_columns=['id', 'name'], missing_strategy='fill')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned, required_columns=['id', 'name', 'age'])
    print(f"\nValidation: {message}")