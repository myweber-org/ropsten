def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    if columns_to_clean is None:
        # Automatically detect string columns
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    # Normalize string columns
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(normalize_string)
    
    return cleaned_df, removed_duplicates

def normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    return text

def validate_email(email):
    """
    Validate email format using regex.
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def main():
    # Example usage
    data = {
        'name': ['John Doe', 'john doe', 'Jane Smith', 'Jane Smith', 'Bob Johnson'],
        'email': ['john@example.com', 'john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net'],
        'age': [25, 25, 30, 30, 35]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df, duplicates_removed = clean_dataframe(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    # Validate emails
    cleaned_df['valid_email'] = cleaned_df['email'].apply(validate_email)
    print("DataFrame with email validation:")
    print(cleaned_df)

if __name__ == "__main__":
    main()