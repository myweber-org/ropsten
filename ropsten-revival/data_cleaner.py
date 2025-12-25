
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
    Clean a list of numeric strings by converting to integers
    and removing any non-numeric entries.
    """
    cleaned = []
    for s in string_list:
        try:
            cleaned.append(int(s))
        except ValueError:
            continue
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5, "6", "7", "abc", "8"]
    print("Original:", sample_data)
    
    # Remove duplicates from the first part
    numeric_part = [x for x in sample_data if isinstance(x, (int, float))]
    unique_numeric = remove_duplicates(numeric_part)
    print("Unique numeric:", unique_numeric)
    
    # Clean string numbers
    string_part = [str(x) for x in sample_data if isinstance(x, str)]
    cleaned_numbers = clean_numeric_strings(string_part)
    print("Cleaned string numbers:", cleaned_numbers)