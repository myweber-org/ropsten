
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data(data):
    """
    Clean data by removing duplicates and filtering out None values.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    
    filtered = [item for item in data if item is not None]
    return remove_duplicates(filtered)

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, None, 4, 3, 5, None]
    cleaned = clean_data(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")