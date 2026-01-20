
import csv
import sys
from typing import Dict, List, Any, Optional

class DataCleaner:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.data = []
        self.headers = []

    def load_data(self) -> None:
        try:
            with open(self.input_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self.headers = reader.fieldnames or []
                self.data = [row for row in reader]
        except FileNotFoundError:
            print(f"Error: File '{self.input_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def handle_missing_values(self, default_values: Optional[Dict[str, Any]] = None) -> None:
        if default_values is None:
            default_values = {}
        
        for row in self.data:
            for header in self.headers:
                if row.get(header) in (None, '', 'NA', 'N/A', 'null'):
                    row[header] = default_values.get(header, 'Unknown')

    def convert_types(self, type_map: Dict[str, str]) -> None:
        for row in self.data:
            for header, target_type in type_map.items():
                if header in row:
                    try:
                        if target_type == 'int':
                            row[header] = int(float(row[header])) if row[header] else 0
                        elif target_type == 'float':
                            row[header] = float(row[header]) if row[header] else 0.0
                        elif target_type == 'bool':
                            row[header] = str(row[header]).lower() in ('true', '1', 'yes', 'y')
                    except (ValueError, TypeError):
                        row[header] = None

    def remove_duplicates(self, key_columns: List[str]) -> None:
        seen = set()
        unique_data = []
        
        for row in self.data:
            key = tuple(row.get(col, '') for col in key_columns)
            if key not in seen:
                seen.add(key)
                unique_data.append(row)
        
        self.data = unique_data

    def save_data(self) -> None:
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(self.data)
            print(f"Cleaned data saved to '{self.output_file}'")
        except Exception as e:
            print(f"Error saving data: {e}")
            sys.exit(1)

    def get_summary(self) -> Dict[str, Any]:
        return {
            'original_rows': len(self.data),
            'columns': self.headers,
            'sample_row': self.data[0] if self.data else {}
        }

def main():
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    cleaner = DataCleaner(input_file, output_file)
    cleaner.load_data()
    
    print(f"Loaded {len(cleaner.data)} rows with {len(cleaner.headers)} columns")
    
    cleaner.handle_missing_values({'age': 0, 'name': 'Unknown'})
    cleaner.convert_types({'age': 'int', 'score': 'float'})
    cleaner.remove_duplicates(['id', 'email'])
    
    cleaner.save_data()
    
    summary = cleaner.get_summary()
    print(f"Processing complete. Final row count: {summary['original_rows']}")

if __name__ == "__main__":
    main()import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing null values and duplicate rows.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
                                          If None, checks all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    if columns_to_check:
        df_cleaned = df_cleaned.drop_duplicates(subset=columns_to_check)
    else:
        df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate that DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 3, 3, 4, None],
#         'name': ['Alice', 'Bob', 'Charlie', 'Charlie', None, 'Eve'],
#         'age': [25, 30, 35, 35, 40, 45]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print(f"Shape: {df.shape}")
#     
#     # Clean the data
#     cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'])
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     print(f"Shape: {cleaned_df.shape}")
#     
#     # Validate
#     is_valid, message = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age'])
#     print(f"\nValidation: {is_valid} - {message}")