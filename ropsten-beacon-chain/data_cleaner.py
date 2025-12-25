import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = df_clean.columns
    
    missing_counts = {}
    for column in columns_to_check:
        if column in df_clean.columns:
            missing_count = df_clean[column].isnull().sum()
            if missing_count > 0:
                missing_counts[column] = missing_count
                # Fill numeric columns with median, others with mode
                if pd.api.types.is_numeric_dtype(df_clean[column]):
                    df_clean[column].fillna(df_clean[column].median(), inplace=True)
                else:
                    df_clean[column].fillna(df_clean[column].mode()[0] if not df_clean[column].mode().empty else '', inplace=True)
    
    # Log cleaning results
    print(f"Removed {removed_duplicates} duplicate rows.")
    if missing_counts:
        print("Missing values handled:")
        for col, count in missing_counts.items():
            print(f"  - {col}: {count} missing values filled")
    else:
        print("No missing values found in specified columns.")
    
    return df_clean

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['id', 'name', 'age'])
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")