import csv
import re
from typing import List, Dict, Any

def remove_duplicates(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    unique_data = []
    for row in data:
        if row[key] not in seen:
            seen.add(row[key])
            unique_data.append(row)
    return unique_data

def normalize_string(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = value.strip()
    value = re.sub(r'\s+', ' ', value)
    return value.lower()

def clean_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

def load_csv(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def save_csv(data: List[Dict[str, Any]], filepath: str, fieldnames: List[str]) -> None:
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def process_csv(input_file: str, output_file: str, unique_key: str) -> None:
    data = load_csv(input_file)
    if not data:
        return
    
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if any(num_term in key.lower() for num_term in ['price', 'amount', 'quantity', 'total']):
                cleaned_row[key] = clean_numeric(value)
            elif isinstance(value, str):
                cleaned_row[key] = normalize_string(value)
            else:
                cleaned_row[key] = value
        cleaned_data.append(cleaned_row)
    
    deduplicated_data = remove_duplicates(cleaned_data, unique_key)
    
    if deduplicated_data:
        fieldnames = list(deduplicated_data[0].keys())
        save_csv(deduplicated_data, output_file, fieldnames)