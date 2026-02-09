def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: List containing potentially duplicate items.
    
    Returns:
        List with duplicates removed.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(data_list):
    """
    Convert string representations of numbers to actual numeric types.
    
    Args:
        data_list: List containing string or numeric values.
    
    Returns:
        List with numeric strings converted to integers or floats.
    """
    cleaned = []
    for item in data_list:
        if isinstance(item, str):
            try:
                if '.' in item:
                    cleaned.append(float(item))
                else:
                    cleaned.append(int(item))
            except ValueError:
                cleaned.append(item)
        else:
            cleaned.append(item)
    return cleaned

def validate_data_types(data_list, expected_type):
    """
    Validate that all items in the list are of the expected type.
    
    Args:
        data_list: List to validate.
        expected_type: Type to check against.
    
    Returns:
        Tuple of (is_valid, invalid_items)
    """
    invalid_items = []
    for item in data_list:
        if not isinstance(item, expected_type):
            invalid_items.append(item)
    
    is_valid = len(invalid_items) == 0
    return is_valid, invalid_itemsimport csv
import re
from typing import List, Dict, Any, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value) if value is not None else ""
    return re.sub(r'\s+', ' ', value.strip())

def parse_numeric(value: str) -> Optional[float]:
    """Convert string to float, handling common separators."""
    if value is None:
        return None
    cleaned = value.replace(',', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))

def clean_csv_row(row: Dict[str, Any], 
                  numeric_fields: List[str] = None,
                  string_fields: List[str] = None) -> Dict[str, Any]:
    """Clean a single CSV row based on field types."""
    cleaned = {}
    
    for key, value in row.items():
        if value is None or str(value).lower() in ['', 'null', 'none']:
            cleaned[key] = None
        elif numeric_fields and key in numeric_fields:
            cleaned[key] = parse_numeric(str(value))
        elif string_fields and key in string_fields:
            cleaned[key] = clean_string(str(value))
        else:
            cleaned[key] = str(value).strip()
    
    return cleaned

def process_csv_file(input_path: str, 
                     output_path: str,
                     numeric_fields: List[str] = None,
                     string_fields: List[str] = None) -> int:
    """Process CSV file and write cleaned version."""
    processed_rows = 0
    
    with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            cleaned_row = clean_csv_row(row, numeric_fields, string_fields)
            writer.writerow(cleaned_row)
            processed_rows += 1
    
    return processed_rows

def get_csv_summary(file_path: str) -> Dict[str, Any]:
    """Generate basic statistics about CSV file."""
    summary = {
        'total_rows': 0,
        'columns': [],
        'sample_data': {}
    }
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            summary['columns'] = reader.fieldnames or []
            
            rows = list(reader)
            summary['total_rows'] = len(rows)
            
            if rows:
                summary['sample_data'] = rows[0]
    
    except FileNotFoundError:
        summary['error'] = f"File not found: {file_path}"
    except Exception as e:
        summary['error'] = f"Processing error: {str(e)}"
    
    return summary