import pandas as pd

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping original column names to new names
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip whitespace, lowercase)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Rename columns if mapping is provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Normalize text columns
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
        print(f"Normalized {len(text_columns)} text columns")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, check_missing=True):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of column names that must be present
        check_missing (bool): Whether to check for missing values
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'column_types': df.dtypes.to_dict()
    }
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_cols
    
    # Check for missing values
    if check_missing:
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df)) * 100
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        validation_results['missing_percentage'] = missing_percentage[missing_counts > 0].to_dict()
    
    return validation_results

def sample_data(df, sample_size=5, random_state=42):
    """
    Return a sample of the DataFrame for inspection.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        sample_size (int): Number of rows to sample
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Sampled DataFrame
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'Alice Brown'],
#         'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com', 'alice@example.com'],
#         'age': [25, 30, 25, 35, 28],
#         'city': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Boston']
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\n" + "="*50 + "\n")
#     
#     # Clean the data
#     cleaned = clean_dataset(df, drop_duplicates=True, normalize_text=True)
#     print("Cleaned DataFrame:")
#     print(cleaned)
#     print("\n" + "="*50 + "\n")
#     
#     # Validate the data
#     validation = validate_data(cleaned, required_columns=['name', 'email', 'age'])
#     print("Validation Results:")
#     for key, value in validation.items():
#         print(f"{key}: {value}")
#     print("\n" + "="*50 + "\n")
#     
#     # Show a sample
#     sample = sample_data(cleaned, sample_size=3)
#     print("Data Sample:")
#     print(sample)