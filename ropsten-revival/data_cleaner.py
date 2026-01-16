
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list (list): Input list potentially containing duplicates.
    
    Returns:
        list: List with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values):
    """
    Clean numeric data by converting strings to floats and removing None values.
    
    Args:
        values (list): List of numeric values as strings or numbers.
    
    Returns:
        list: Cleaned list of float values.
    """
    cleaned = []
    
    for val in values:
        if val is None:
            continue
        
        try:
            cleaned.append(float(val))
        except (ValueError, TypeError):
            continue
    
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned_data = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned_data}")
    
    numeric_data = ["1.5", "2.3", None, "invalid", "3.7"]
    cleaned_numeric = clean_numeric_data(numeric_data)
    print(f"Numeric original: {numeric_data}")
    print(f"Numeric cleaned: {cleaned_numeric}")