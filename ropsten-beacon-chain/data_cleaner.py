
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

def clean_numeric_strings(string_list):
    """
    Clean a list of strings by converting numeric strings to integers.
    Non-numeric strings are kept as-is.
    """
    cleaned = []
    for s in string_list:
        if isinstance(s, str) and s.isdigit():
            cleaned.append(int(s))
        else:
            cleaned.append(s)
    return cleaned

def filter_by_type(data_list, target_type):
    """
    Filter a list to include only elements of a specific type.
    """
    return [item for item in data_list if isinstance(item, target_type)]

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, "4", "4", "hello", 5.5, 5.5]
    print("Original:", sample_data)
    print("No duplicates:", remove_duplicates(sample_data))
    print("Cleaned numeric strings:", clean_numeric_strings(sample_data))
    print("Integers only:", filter_by_type(sample_data, int))