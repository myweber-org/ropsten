import csv
import os

def read_csv_file(file_path):
    """Read a CSV file and return its data as a list of dictionaries."""
    data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return data

def clean_numeric_fields(data, fields):
    """Clean specified numeric fields by removing non-numeric characters and converting to float."""
    cleaned_data = []
    for row in data:
        cleaned_row = row.copy()
        for field in fields:
            if field in cleaned_row:
                value = cleaned_row[field]
                if isinstance(value, str):
                    cleaned_value = ''.join(char for char in value if char.isdigit() or char == '.')
                    try:
                        cleaned_row[field] = float(cleaned_value) if cleaned_value else 0.0
                    except ValueError:
                        cleaned_row[field] = 0.0
        cleaned_data.append(cleaned_row)
    return cleaned_data

def write_csv_file(data, file_path):
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Data successfully written to '{file_path}'.")
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def main():
    input_file = "input_data.csv"
    output_file = "cleaned_data.csv"
    numeric_fields = ["price", "quantity", "score"]

    data = read_csv_file(input_file)
    if data:
        cleaned_data = clean_numeric_fields(data, numeric_fields)
        write_csv_file(cleaned_data, output_file)

if __name__ == "__main__":
    main()