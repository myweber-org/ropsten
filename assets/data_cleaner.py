import csv
import re
from typing import List, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value)
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def read_csv_file(filepath: str) -> List[dict]:
    """Read CSV file and return list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return data

def clean_csv_data(data: List[dict], email_field: Optional[str] = None) -> List[dict]:
    """Clean CSV data by processing string fields and optionally validating emails."""
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if isinstance(value, str):
                cleaned_value = clean_string(value)
                if email_field and key == email_field:
                    if not validate_email(cleaned_value):
                        print(f"Warning: Invalid email '{cleaned_value}' in field '{key}'")
                cleaned_row[key] = cleaned_value
            else:
                cleaned_row[key] = value
        cleaned_data.append(cleaned_row)
    return cleaned_data

def write_csv_file(filepath: str, data: List[dict], fieldnames: Optional[List[str]] = None) -> bool:
    """Write data to CSV file."""
    if not data:
        print("Error: No data to write.")
        return False
    
    if not fieldnames:
        fieldnames = list(data[0].keys())
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False

def process_csv(input_file: str, output_file: str, email_field: Optional[str] = None) -> None:
    """Main function to process CSV file."""
    print(f"Processing {input_file}...")
    data = read_csv_file(input_file)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    cleaned_data = clean_csv_data(data, email_field)
    
    if write_csv_file(output_file, cleaned_data):
        print(f"Cleaned data written to {output_file}")
    else:
        print("Failed to write output file.")

if __name__ == "__main__":
    input_csv = "input_data.csv"
    output_csv = "cleaned_data.csv"
    process_csv(input_csv, output_csv, email_field="email")
def remove_duplicates_preserve_order(sequence):
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
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    if columns_to_clean is None:
        # Automatically detect string columns
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].apply(_normalize_string)
    
    return df_cleaned

def _normalize_string(text):
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
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def validate_email(email):
    """
    Validate email format using regex.
    """
    if pd.isna(email):
        return False
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, str(email)))

def clean_phone_number(phone):
    """
    Clean and format phone numbers to a standard format.
    """
    if pd.isna(phone):
        return phone
    
    phone = str(phone)
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    else:
        return phone