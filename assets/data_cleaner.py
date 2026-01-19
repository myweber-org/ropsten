import re
from typing import List, Dict, Any

def remove_duplicates(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    unique_data = []
    for item in data:
        if item.get(key) not in seen:
            seen.add(item.get(key))
            unique_data.append(item)
    return unique_data

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_data(data: List[Dict[str, Any]], text_fields: List[str]) -> List[Dict[str, Any]]:
    cleaned = []
    for item in data:
        cleaned_item = item.copy()
        for field in text_fields:
            if field in cleaned_item and isinstance(cleaned_item[field], str):
                cleaned_item[field] = normalize_text(cleaned_item[field])
        cleaned.append(cleaned_item)
    return cleaned

def process_dataset(data: List[Dict[str, Any]], unique_key: str, text_fields: List[str]) -> List[Dict[str, Any]]:
    deduplicated = remove_duplicates(data, unique_key)
    cleaned = clean_data(deduplicated, text_fields)
    return cleaned
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result