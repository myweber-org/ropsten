
import csv
import sys

def remove_duplicates(input_file, output_file, key_columns):
    """
    Remove duplicate rows from a CSV file based on specified key columns.
    Keep the first occurrence of each duplicate.
    """
    seen = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Create a tuple of values from the key columns
            key = tuple(row[col] for col in key_columns)
            
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    
    return len(unique_rows)

def main():
    if len(sys.argv) < 4:
        print("Usage: python data_cleaner.py <input_file> <output_file> <key_column1> [key_column2 ...]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key_columns = sys.argv[3:]
    
    try:
        count = remove_duplicates(input_file, output_file, key_columns)
        print(f"Processed {count} unique rows. Output saved to {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except KeyError as e:
        print(f"Error: Key column {e} not found in CSV header.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()