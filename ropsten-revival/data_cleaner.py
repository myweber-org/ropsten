
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
    
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if all required columns are present
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def main():
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'age': ['25', '30', '30', '35', '40', '40'],
        'score': ['85.5', '90.0', '90.0', '78.5', '92.0', '92.0']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Remove duplicates
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    print("After removing duplicates:")
    print(df_clean)
    print()
    
    # Clean numeric columns
    df_clean = clean_numeric_columns(df_clean, ['age', 'score'])
    print("After cleaning numeric columns:")
    print(df_clean)
    print()
    
    # Validate required columns
    required = ['id', 'name', 'age']
    is_valid = validate_dataframe(df_clean, required)
    print(f"DataFrame validation: {is_valid}")

if __name__ == "__main__":
    main()