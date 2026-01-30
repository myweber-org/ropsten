
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list containing elements that may have duplicates.
    
    Returns:
        A new list with duplicates removed, preserving the original order.
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_data_with_threshold(data_list, threshold=None):
    """
    Clean data by removing duplicates, optionally with a frequency threshold.
    
    Args:
        data_list: List of data elements to clean.
        threshold: Optional integer threshold. If provided, only elements
                  appearing more than threshold times are considered duplicates.
    
    Returns:
        Cleaned list with duplicates removed.
    """
    if threshold is None:
        return remove_duplicates(data_list)
    
    from collections import Counter
    counter = Counter(data_list)
    
    result = []
    seen = set()
    
    for item in data_list:
        if counter[item] > threshold:
            if item not in seen:
                seen.add(item)
                result.append(item)
        else:
            result.append(item)
    
    return result

def validate_input(data):
    """
    Validate that input is a list or convertible to list.
    
    Args:
        data: Input data to validate.
    
    Returns:
        Validated list.
    
    Raises:
        TypeError: If input cannot be converted to list.
    """
    if isinstance(data, list):
        return data
    elif hasattr(data, '__iter__'):
        return list(data)
    else:
        raise TypeError("Input must be iterable")

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1, 6]
    
    print("Original data:", sample_data)
    print("After basic deduplication:", remove_duplicates(sample_data))
    print("With threshold 2:", clean_data_with_threshold(sample_data, threshold=2))
    
    # Test validation
    try:
        validate_input("not a list")
    except TypeError as e:
        print(f"Validation error: {e}")
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result