
import csv
import json
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read CSV file and return list of dictionaries."""
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return [row for row in reader]
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []

def validate_row(row: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate that row contains all required fields with non-empty values."""
    for field in required_fields:
        if field not in row or not str(row[field]).strip():
            return False
    return True

def transform_numeric_fields(rows: List[Dict[str, Any]], numeric_fields: List[str]) -> List[Dict[str, Any]]:
    """Convert specified fields to numeric values, handling invalid data."""
    transformed = []
    for row in rows:
        new_row = row.copy()
        for field in numeric_fields:
            if field in new_row:
                try:
                    new_row[field] = float(new_row[field])
                except (ValueError, TypeError):
                    new_row[field] = None
        transformed.append(new_row)
    return transformed

def calculate_statistics(rows: List[Dict[str, Any]], field: str) -> Dict[str, Optional[float]]:
    """Calculate basic statistics for a numeric field."""
    values = [row.get(field) for row in rows if isinstance(row.get(field), (int, float))]
    
    if not values:
        return {"min": None, "max": None, "avg": None, "count": 0}
    
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "count": len(values)
    }

def filter_rows(rows: List[Dict[str, Any]], condition_func) -> List[Dict[str, Any]]:
    """Filter rows based on a condition function."""
    return [row for row in rows if condition_func(row)]

def export_to_json(data: List[Dict[str, Any]], output_path: str) -> bool:
    """Export data to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2, ensure_ascii=False)
        logger.info(f"Data exported to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        return False

def process_csv_pipeline(input_file: str, output_file: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Complete CSV processing pipeline with validation and transformation."""
    logger.info(f"Starting CSV processing pipeline for {input_file}")
    
    # Read CSV data
    raw_data = read_csv_file(input_file)
    if not raw_data:
        return {"success": False, "message": "No data read from CSV file"}
    
    # Validate data
    required_fields = config.get("required_fields", [])
    valid_data = [row for row in raw_data if validate_row(row, required_fields)]
    
    if len(valid_data) < len(raw_data):
        logger.warning(f"Filtered out {len(raw_data) - len(valid_data)} invalid rows")
    
    # Transform numeric fields
    numeric_fields = config.get("numeric_fields", [])
    transformed_data = transform_numeric_fields(valid_data, numeric_fields)
    
    # Apply filters if specified
    if "filter_condition" in config:
        filtered_data = filter_rows(transformed_data, config["filter_condition"])
    else:
        filtered_data = transformed_data
    
    # Calculate statistics if requested
    statistics = {}
    if "statistics_fields" in config:
        for field in config["statistics_fields"]:
            statistics[field] = calculate_statistics(filtered_data, field)
    
    # Export results
    export_success = export_to_json(filtered_data, output_file)
    
    result = {
        "success": export_success,
        "input_file": input_file,
        "output_file": output_file,
        "original_rows": len(raw_data),
        "valid_rows": len(valid_data),
        "processed_rows": len(filtered_data),
        "statistics": statistics
    }
    
    logger.info(f"Processing complete: {result}")
    return result

def example_usage():
    """Example usage of the CSV processing functions."""
    
    # Example configuration
    config = {
        "required_fields": ["id", "name", "value"],
        "numeric_fields": ["value", "score"],
        "statistics_fields": ["value"],
        "filter_condition": lambda row: row.get("value", 0) > 50
    }
    
    # Process the CSV file
    result = process_csv_pipeline(
        input_file="data/input.csv",
        output_file="data/output.json",
        config=config
    )
    
    return result

if __name__ == "__main__":
    # Run example if script is executed directly
    example_result = example_usage()
    print(f"Processing result: {example_result}")