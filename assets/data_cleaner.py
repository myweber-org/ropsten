import csv
import re
from typing import List, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and convert to lowercase."""
    if not isinstance(value, str):
        return str(value)
    return re.sub(r'\s+', ' ', value.strip()).lower()

def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def read_csv_file(filepath: str) -> List[dict]:
    """Read CSV file and return list of dictionaries."""
    data = []
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return data

def clean_csv_data(data: List[dict]) -> List[dict]:
    """Clean all string fields in CSV data."""
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if isinstance(value, str):
                cleaned_row[key] = clean_string(value)
            else:
                cleaned_row[key] = value
        cleaned_data.append(cleaned_row)
    return cleaned_data

def filter_valid_emails(data: List[dict], email_field: str = 'email') -> List[dict]:
    """Filter rows with valid email addresses."""
    valid_data = []
    for row in data:
        email = row.get(email_field)
        if email and validate_email(email):
            valid_data.append(row)
    return valid_data

def write_csv_file(filepath: str, data: List[dict], fieldnames: Optional[List[str]] = None):
    """Write data to CSV file."""
    if not data:
        print("No data to write.")
        return
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    try:
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Data successfully written to '{filepath}'")
    except Exception as e:
        print(f"Error writing CSV: {e}")

def process_csv(input_file: str, output_file: str):
    """Main function to process CSV file."""
    print(f"Processing CSV file: {input_file}")
    
    raw_data = read_csv_file(input_file)
    if not raw_data:
        return
    
    print(f"Read {len(raw_data)} rows from CSV")
    
    cleaned_data = clean_csv_data(raw_data)
    valid_data = filter_valid_emails(cleaned_data)
    
    print(f"Found {len(valid_data)} rows with valid emails")
    
    write_csv_file(output_file, valid_data)
    
    print(f"Processing complete. Output saved to: {output_file}")

if __name__ == "__main__":
    process_csv("input_data.csv", "cleaned_data.csv")