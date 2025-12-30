
import pandas as pd

def clean_dataframe(df, column_to_deduplicate, columns_to_normalize=None):
    """
    Clean a pandas DataFrame by removing duplicates from a specified column
    and normalizing string columns to lowercase and stripping whitespace.
    """
    # Remove duplicates based on the specified column
    df_cleaned = df.drop_duplicates(subset=[column_to_deduplicate], keep='first')
    
    # Normalize specified string columns
    if columns_to_normalize:
        for col in columns_to_normalize:
            if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].astype(str).str.lower().str.strip()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    Returns a boolean and a list of missing columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'user_id': [1, 2, 3, 2, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'bob', 'David'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'bob@example.com', 'david@example.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    cleaned_df = clean_dataframe(df, 'user_id', columns_to_normalize=['name', 'email'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    # Validate the cleaned data
    required_cols = ['user_id', 'name', 'email']
    is_valid, missing = validate_dataframe(cleaned_df, required_cols)
    print(f"Data validation passed: {is_valid}")
    if not is_valid:
        print(f"Missing columns: {missing}")
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result