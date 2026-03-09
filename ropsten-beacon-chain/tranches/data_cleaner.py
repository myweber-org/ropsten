
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_list = [3, 1, 2, 1, 4, 3, 5, 2]
    cleaned = remove_duplicates_preserve_order(sample_list)
    print(f"Original: {sample_list}")
    print(f"Cleaned: {cleaned}")
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    if columns_to_clean is None:
        # Automatically detect string columns
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    for column in columns_to_clean:
        if column in df_cleaned.columns:
            df_cleaned[column] = df_cleaned[column].apply(normalize_string)
    
    return df_cleaned

def normalize_string(value):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(value):
        return value
    
    if isinstance(value, str):
        # Convert to lowercase
        normalized = value.lower()
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        # Remove special characters (keep alphanumeric and spaces)
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        return normalized
    
    return value

def validate_email(email):
    """
    Validate email format using regex pattern.
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def sample_usage():
    """
    Demonstrate usage of the data cleaning functions.
    """
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Alice Johnson  '],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'alice@example.com'],
        'age': [25, 30, 25, 35]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    # Validate emails
    cleaned_df['valid_email'] = cleaned_df['email'].apply(validate_email)
    print("DataFrame with email validation:")
    print(cleaned_df)

if __name__ == "__main__":
    sample_usage()