
import pandas as pd

def clean_dataset(df, sort_column=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and optionally sorting.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        sort_column (str, optional): Column name to sort by. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and sorted if specified.
    """
    cleaned_df = df.drop_duplicates().reset_index(drop=True)
    
    if sort_column and sort_column in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(by=sort_column).reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Warning: DataFrame is empty.")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': [85, 92, 92, 78, 88, 88, 95]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned = clean_dataset(df, sort_column='score')
    print("Cleaned DataFrame (sorted by score):")
    print(cleaned)
    print()
    
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'name', 'score'])
    print(f"Data validation passed: {is_valid}")