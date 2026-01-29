import csv
import re
from typing import List, Dict, Any, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and convert to lowercase."""
    if not isinstance(value, str):
        return str(value)
    return re.sub(r'\s+', ' ', value.strip()).lower()

def clean_numeric(value: str) -> Optional[float]:
    """Convert string to float, handling common formatting issues."""
    if not value:
        return None
    cleaned = value.replace(',', '').replace('$', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

def read_and_clean_csv(filepath: str) -> List[Dict[str, Any]]:
    """Read CSV file and apply cleaning functions to each row."""
    cleaned_data = []
    
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                if any(num_term in key.lower() for num_term in ['price', 'amount', 'quantity', 'total']):
                    cleaned_row[key] = clean_numeric(value)
                else:
                    cleaned_row[key] = clean_string(value)
            cleaned_data.append(cleaned_row)
    
    return cleaned_data

def write_cleaned_csv(data: List[Dict[str, Any]], output_path: str) -> None:
    """Write cleaned data to a new CSV file."""
    if not data:
        return
    
    fieldnames = data[0].keys()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def validate_email(email: str) -> bool:
    """Basic email validation using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))

def remove_duplicates(data: List[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
    """Remove duplicate rows based on a specified key field."""
    seen = set()
    unique_data = []
    
    for row in data:
        key_value = row.get(key_field)
        if key_value not in seen:
            seen.add(key_value)
            unique_data.append(row)
    
    return unique_data

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        raw_data = read_and_clean_csv(input_file)
        unique_data = remove_duplicates(raw_data, "id")
        write_cleaned_csv(unique_data, output_file)
        print(f"cleaned {len(unique_data)} records")
    except FileNotFoundError:
        print(f"error: file '{input_file}' not found")
    except Exception as e:
        print(f"error processing file: {e}")