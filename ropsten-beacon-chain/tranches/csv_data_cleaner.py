import csv
import sys

def clean_csv(input_file, output_file):
    seen = set()
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        
        for row in reader:
            # Trim whitespace from each field
            trimmed_row = [field.strip() for field in row]
            row_tuple = tuple(trimmed_row)
            
            if row_tuple not in seen:
                seen.add(row_tuple)
                cleaned_rows.append(trimmed_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(cleaned_rows)
    
    print(f"Cleaned data saved to {output_file}")
    print(f"Removed {len(seen) - len(cleaned_rows)} duplicate rows")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    clean_csv(sys.argv[1], sys.argv[2])import csv
import sys

def clean_csv(input_file, output_file):
    """
    Clean a CSV file by removing rows where all fields are empty
    and trimming whitespace from all string fields.
    """
    cleaned_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                # Skip rows where all fields are empty or whitespace
                if all(field.strip() == '' for field in row):
                    continue
                
                # Trim whitespace from each field
                cleaned_row = [field.strip() if isinstance(field, str) else field for field in row]
                cleaned_rows.append(cleaned_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(cleaned_rows)
            
        print(f"Successfully cleaned {len(cleaned_rows)} rows. Output saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    clean_csv(input_path, output_path)