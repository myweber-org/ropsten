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