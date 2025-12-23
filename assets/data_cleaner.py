
import pandas as pd
import numpy as np

def clean_dataframe(df, text_columns=None):
    """
    Clean a DataFrame by removing null values and standardizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        text_columns (list): List of column names containing text data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove rows with any null values
    cleaned_df = cleaned_df.dropna()
    
    # Standardize text columns if specified
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                # Convert to string, strip whitespace, and convert to lowercase
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
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
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def main():
    # Example usage
    sample_data = {
        'name': ['John Doe', 'Jane Smith', None, 'Bob Johnson', '   ALICE   '],
        'age': [25, 30, 35, None, 28],
        'email': ['john@example.com', 'JANE@EXAMPLE.COM', 'bob@test.com', None, 'alice@demo.org']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataframe(df, text_columns=['name', 'email'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['name', 'age', 'email'])
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")

if __name__ == "__main__":
    main()