
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str or dict): Method to fill missing values. 
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
                                If None, missing values are not filled.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            for column, value in fill_missing.items():
                if column in cleaned_df.columns:
                    cleaned_df[column].fillna(value, inplace=True)
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'mode':
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype == 'object':
                    mode_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else None
                    cleaned_df[column].fillna(mode_value, inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows, but has {len(df)}"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for missing values: 'mean', 'median', 'mode', or 'drop'
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif strategy == 'mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    elif strategy == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std())
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        cleaned_df = cleaned_df[outlier_mask]
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
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
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 8],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, strategy='mean', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=2)
    print(f"\nValidation: {is_valid} - {message}")