
import sys

def remove_duplicates(input_file, output_file):
    seen = set()
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line not in seen:
                seen.add(line)
                outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        remove_duplicates(input_path, output_path)
        print(f"Duplicates removed. Cleaned file saved as: {output_path}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")