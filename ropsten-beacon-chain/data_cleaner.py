def remove_duplicates(input_list):
    """
    Removes duplicate elements from a list while preserving the original order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary for renaming columns (optional)
        drop_duplicates: Boolean flag to remove duplicate rows
        fill_missing: Boolean flag to fill missing values
    
    Returns:
        Cleaned pandas DataFrame
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
        removed_duplicates = initial_rows - len(cleaned_df)
        print(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    if fill_missing:
        # Fill numeric columns with median
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                print(f"Filled missing values in '{col}' with median: {median_val}")
        
        # Fill categorical columns with mode
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown'
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                print(f"Filled missing values in '{col}' with mode: {mode_val}")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df: Input pandas DataFrame
        required_columns: List of column names that must be present
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
        validation_results['all_required_columns_present'] = len(missing_cols) == 0
    
    return validation_results

# Example usage function
def process_sample_data():
    """Demonstrate the data cleaning functionality with sample data."""
    # Create sample data with issues
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', None, 'Eve'],
        'age': [25, 30, None, 35, 28, 32],
        'score': [85.5, 92.0, 78.5, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing=True)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned, required_columns=['id', 'name', 'age', 'score'])
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    return cleaned

if __name__ == "__main__":
    # Run the example when script is executed directly
    cleaned_data = process_sample_data()