
import csv
import json
from typing import List, Dict, Any, Optional

def read_csv_file(file_path: str) -> List[Dict[str, str]]:
    """Read CSV file and return list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return data

def validate_row(row: Dict[str, str], required_fields: List[str]) -> bool:
    """Validate if row contains all required fields with non-empty values."""
    for field in required_fields:
        if field not in row or not row[field].strip():
            return False
    return True

def transform_numeric_fields(row: Dict[str, str], numeric_fields: List[str]) -> Dict[str, Any]:
    """Convert specified fields to numeric types if possible."""
    transformed = row.copy()
    for field in numeric_fields:
        if field in transformed:
            try:
                value = transformed[field].strip()
                if '.' in value:
                    transformed[field] = float(value)
                else:
                    transformed[field] = int(value)
            except (ValueError, TypeError):
                transformed[field] = None
    return transformed

def filter_data(data: List[Dict[str, Any]], filter_func) -> List[Dict[str, Any]]:
    """Filter data using provided filter function."""
    return [row for row in data if filter_func(row)]

def calculate_statistics(data: List[Dict[str, Any]], field: str) -> Dict[str, Optional[float]]:
    """Calculate basic statistics for a numeric field."""
    values = []
    for row in data:
        if field in row and isinstance(row[field], (int, float)):
            values.append(row[field])
    
    if not values:
        return {"min": None, "max": None, "avg": None, "count": 0}
    
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "count": len(values)
    }

def export_to_json(data: List[Dict[str, Any]], output_path: str) -> bool:
    """Export processed data to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error exporting to JSON: {e}")
        return False

def process_csv_pipeline(input_file: str, output_file: str, 
                         required_fields: List[str],
                         numeric_fields: List[str]) -> Dict[str, Any]:
    """Main pipeline for CSV data processing."""
    result = {
        "total_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "statistics": {},
        "export_success": False
    }
    
    raw_data = read_csv_file(input_file)
    result["total_rows"] = len(raw_data)
    
    if not raw_data:
        return result
    
    processed_data = []
    for row in raw_data:
        if validate_row(row, required_fields):
            transformed_row = transform_numeric_fields(row, numeric_fields)
            processed_data.append(transformed_row)
            result["valid_rows"] += 1
        else:
            result["invalid_rows"] += 1
    
    for field in numeric_fields:
        result["statistics"][field] = calculate_statistics(processed_data, field)
    
    if processed_data:
        result["export_success"] = export_to_json(processed_data, output_file)
    
    return result

if __name__ == "__main__":
    sample_data = [
        {"name": "Alice", "age": "25", "score": "95.5"},
        {"name": "Bob", "age": "30", "score": "88.0"},
        {"name": "Charlie", "age": "", "score": "92.5"}
    ]
    
    with open('sample.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age", "score"])
        writer.writeheader()
        writer.writerows(sample_data)
    
    result = process_csv_pipeline(
        input_file='sample.csv',
        output_file='processed_data.json',
        required_fields=['name', 'age'],
        numeric_fields=['age', 'score']
    )
    
    print(f"Processing complete. Result: {json.dumps(result, indent=2)}")