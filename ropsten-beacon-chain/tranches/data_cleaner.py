def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: List containing potentially duplicate items.
    
    Returns:
        List with duplicates removed.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(data_list):
    """
    Convert string representations of numbers to actual numeric types.
    
    Args:
        data_list: List containing string or numeric values.
    
    Returns:
        List with numeric strings converted to integers or floats.
    """
    cleaned = []
    for item in data_list:
        if isinstance(item, str):
            try:
                if '.' in item:
                    cleaned.append(float(item))
                else:
                    cleaned.append(int(item))
            except ValueError:
                cleaned.append(item)
        else:
            cleaned.append(item)
    return cleaned

def validate_data_types(data_list, expected_type):
    """
    Validate that all items in the list are of the expected type.
    
    Args:
        data_list: List to validate.
        expected_type: Type to check against.
    
    Returns:
        Tuple of (is_valid, invalid_items)
    """
    invalid_items = []
    for item in data_list:
        if not isinstance(item, expected_type):
            invalid_items.append(item)
    
    is_valid = len(invalid_items) == 0
    return is_valid, invalid_items