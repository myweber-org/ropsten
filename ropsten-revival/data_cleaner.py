
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_method: Method to fill missing values ('mean', 'median', 'mode', or None)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_method == 'mean':
            for col in numeric_cols:
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
        elif fill_method == 'median':
            for col in numeric_cols:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        elif fill_method == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    # Remove duplicates
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - len(cleaned_df)
        if removed > 0:
            print(f"Removed {removed} duplicate row(s)")
    
    # Report cleaning summary
    null_count = cleaned_df.isnull().sum().sum()
    if null_count > 0:
        print(f"Warning: {null_count} missing values remain in the dataset")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, None, 15.0, 20.0, None, 30.0],
        'category': ['A', 'B', 'B', 'A', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_method='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)