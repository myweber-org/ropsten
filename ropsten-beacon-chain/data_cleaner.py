import pandas as pd
import numpy as np

def load_and_clean_csv(filepath, drop_na=True, fill_strategy='mean'):
    """
    Load a CSV file and perform basic cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: File is empty")
        return None

    original_shape = df.shape
    print(f"Original data shape: {original_shape}")

    if drop_na:
        df_cleaned = df.dropna()
        print(f"Removed {original_shape[0] - df_cleaned.shape[0]} rows with missing values")
    else:
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if fill_strategy == 'mean':
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
        elif fill_strategy == 'median':
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
        elif fill_strategy == 'zero':
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(0)
        print(f"Filled missing values using {fill_strategy} strategy")

    duplicate_count = df_cleaned.duplicated().sum()
    if duplicate_count > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"Removed {duplicate_count} duplicate rows")

    final_shape = df_cleaned.shape
    print(f"Cleaned data shape: {final_shape}")
    print(f"Total rows removed: {original_shape[0] - final_shape[0]}")

    return df_cleaned

def validate_numeric_range(df, column_name, min_val=None, max_val=None):
    """
    Validate that values in a numeric column are within specified range.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame")
        return False

    if not np.issubdtype(df[column_name].dtype, np.number):
        print(f"Error: Column '{column_name}' is not numeric")
        return False

    violations = 0
    if min_val is not None:
        below_min = (df[column_name] < min_val).sum()
        violations += below_min
        if below_min > 0:
            print(f"Found {below_min} values below minimum {min_val}")

    if max_val is not None:
        above_max = (df[column_name] > max_val).sum()
        violations += above_max
        if above_max > 0:
            print(f"Found {above_max} values above maximum {max_val}")

    if violations == 0:
        print(f"All values in '{column_name}' are within valid range")
        return True
    else:
        print(f"Total range violations in '{column_name}': {violations}")
        return False

def remove_outliers_iqr(df, column_name, multiplier=1.5):
    """
    Remove outliers from a numeric column using IQR method.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found")
        return df

    if not np.issubdtype(df[column_name].dtype, np.number):
        print(f"Error: Column '{column_name}' is not numeric")
        return df

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    original_count = len(df)
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    removed_count = original_count - len(df_filtered)

    if removed_count > 0:
        print(f"Removed {removed_count} outliers from '{column_name}' using IQR method")
        print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    else:
        print(f"No outliers detected in '{column_name}'")

    return df_filtered

def save_cleaned_data(df, output_path, index=False):
    """
    Save cleaned DataFrame to CSV file.
    """
    try:
        df.to_csv(output_path, index=index)
        print(f"Cleaned data saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_string_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize string columns by converting to lowercase and stripping whitespace.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized string columns
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
    return df_copy

def clean_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill with value
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        Cleaned DataFrame
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill' and fill_value is not None:
        return df.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy or missing fill_value")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        True if all required columns are present
    """
    return all(col in df.columns for col in required_columns)

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', ' Bob Johnson '],
        'age': [25, 30, 25, 35],
        'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    df_no_duplicates = remove_duplicate_rows(df, subset=['email'])
    print("After removing duplicates by email:")
    print(df_no_duplicates)
    print("\n")
    
    df_normalized = normalize_string_columns(df_no_duplicates, ['name'])
    print("After normalizing name column:")
    print(df_normalized)
    print("\n")
    
    is_valid = validate_dataframe(df_normalized, ['name', 'age', 'email'])
    print(f"DataFrame validation: {is_valid}")

if __name__ == "__main__":
    example_usage()