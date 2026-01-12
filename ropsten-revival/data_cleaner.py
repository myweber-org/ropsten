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