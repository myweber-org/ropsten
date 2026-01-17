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
    Clean a list of strings by removing non-numeric characters
    and converting to integers where possible.
    """
    cleaned = []
    for s in string_list:
        try:
            numeric_part = ''.join(filter(str.isdigit, s))
            if numeric_part:
                cleaned.append(int(numeric_part))
        except ValueError:
            continue
    return cleaned

def validate_email_list(email_list):
    """
    Basic email validation for a list of email addresses.
    Returns only emails containing '@' and '.' characters.
    """
    valid_emails = []
    for email in email_list:
        if '@' in email and '.' in email.split('@')[-1]:
            valid_emails.append(email.strip().lower())
    return valid_emails

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    sample_strings = ["abc123", "456def", "789", "xyz"]
    print("Numeric cleaned:", clean_numeric_strings(sample_strings))
    
    emails = ["test@example.com", "invalid", "user@domain.org"]
    print("Valid emails:", validate_email_list(emails))