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
    
    clean_csv(sys.argv[1], sys.argv[2])