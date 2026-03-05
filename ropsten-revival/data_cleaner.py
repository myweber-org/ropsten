
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
                                          If None, checks all columns.
        fill_missing (bool): If True, fill missing values with column mean for numeric
                            columns and mode for categorical columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    # Remove duplicates
    if columns_to_check:
        df_clean = df.drop_duplicates(subset=columns_to_check)
    else:
        df_clean = df.drop_duplicates()
    
    duplicates_removed = original_shape[0] - df_clean.shape[0]
    
    # Handle missing values if requested
    if fill_missing:
        for column in df_clean.columns:
            if df_clean[column].dtype in ['int64', 'float64']:
                # Fill numeric columns with mean
                mean_value = df_clean[column].mean()
                df_clean[column] = df_clean[column].fillna(mean_value)
            else:
                # Fill categorical columns with mode
                mode_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else 'Unknown'
                df_clean[column] = df_clean[column].fillna(mode_value)
    
    missing_filled = df.isna().sum().sum() - df_clean.isna().sum().sum()
    
    print(f"Removed {duplicates_removed} duplicate rows")
    print(f"Filled {missing_filled} missing values")
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df_clean.shape}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        bool: True if DataFrame passes validation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'Eve', 'Eve'],
        'age': [25, 30, 30, None, 35, 28, 28],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0, 88.0],
        'department': ['HR', 'IT', 'IT', 'Finance', 'IT', 'HR', 'HR']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing=True)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)