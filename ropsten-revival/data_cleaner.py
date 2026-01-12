import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=np.nan):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df, fill_value=0)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age'])
        print("DataFrame validation passed")
    except ValueError as e:
        print(f"Validation error: {e}")

if __name__ == "__main__":
    main()
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates and optionally filtering by frequency threshold.
    If threshold is provided, only items appearing at least threshold times are kept.
    """
    if not data:
        return []
    
    # Count frequencies
    frequency = {}
    for item in data:
        frequency[item] = frequency.get(item, 0) + 1
    
    # Apply threshold if specified
    if threshold is not None:
        filtered_items = [item for item in data if frequency[item] >= threshold]
    else:
        filtered_items = data
    
    # Remove duplicates while preserving order
    return remove_duplicates(filtered_items)

def validate_data_types(data, expected_type):
    """
    Validate that all elements in the data list are of the expected type.
    Returns a tuple of (is_valid, invalid_indices).
    """
    invalid_indices = []
    for idx, item in enumerate(data):
        if not isinstance(item, expected_type):
            invalid_indices.append(idx)
    
    return (len(invalid_indices) == 0, invalid_indices)

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, "a", "a", "b"]
    
    print("Original data:", sample_data)
    print("Cleaned data:", remove_duplicates(sample_data))
    print("Cleaned with threshold 2:", clean_data_with_threshold(sample_data, threshold=2))
    
    is_valid, invalid_idx = validate_data_types(sample_data, int)
    print(f"All integers? {is_valid}")
    if not is_valid:
        print(f"Invalid indices: {invalid_idx}")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for duplicates
    keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = len(df) - len(cleaned_df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by removing non-numeric characters and converting to float.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(
                cleaned_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                errors='coerce'
            )
    
    return cleaned_df

def standardize_text(df, columns):
    """
    Standardize text columns by converting to lowercase and stripping whitespace.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized text columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
    
    return cleaned_df

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('drop', 'fill')
    fill_value: Value to use for filling missing values
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if strategy == 'drop':
        cleaned_df = df.dropna()
        print(f"Dropped {len(df) - len(cleaned_df)} rows with missing values")
    elif strategy == 'fill':
        cleaned_df = df.fillna(fill_value)
        print(f"Filled missing values with {fill_value}")
    else:
        cleaned_df = df.copy()
    
    return cleaned_df.reset_index(drop=True)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"