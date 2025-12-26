import csv
import re

def clean_csv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            cleaned_row = []
            for cell in row:
                cleaned_cell = re.sub(r'\s+', ' ', cell.strip())
                cleaned_cell = cleaned_cell.replace('"', "'")
                cleaned_row.append(cleaned_cell)
            writer.writerow(cleaned_row)

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_duplicates(input_file, output_file):
    seen = set()
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            row_tuple = tuple(row)
            if row_tuple not in seen:
                seen.add(row_tuple)
                writer.writerow(row)

if __name__ == "__main__":
    clean_csv("raw_data.csv", "cleaned_data.csv")
    print("Data cleaning completed.")