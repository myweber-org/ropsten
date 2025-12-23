
import argparse
import sys

def remove_duplicates(input_file, output_file):
    seen = set()
    try:
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
        
        with open(output_file, 'w') as outfile:
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    outfile.write(line)
        
        print(f"Successfully removed duplicates. Original lines: {len(lines)}, Unique lines: {len(seen)}")
        return True
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except IOError as e:
        print(f"Error processing files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate lines from a text file.')
    parser.add_argument('input', help='Path to the input file')
    parser.add_argument('output', help='Path to the output file')
    
    args = parser.parse_args()
    
    if not remove_duplicates(args.input, args.output):
        sys.exit(1)

if __name__ == "__main__":
    main()import os
import re
import sys

def normalize_filename(filename):
    """Convert filename to lowercase, replace spaces with underscores, and remove special chars."""
    name, ext = os.path.splitext(filename)
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name + ext.lower()

def clean_directory(directory_path):
    """Rename all files in the directory with normalized names."""
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return False

    for filename in os.listdir(directory_path):
        old_path = os.path.join(directory_path, filename)
        if os.path.isfile(old_path):
            new_name = normalize_filename(filename)
            new_path = os.path.join(directory_path, new_name)
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                except OSError as e:
                    print(f"Failed to rename {filename}: {e}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_cleaner.py <directory_path>")
        sys.exit(1)
    target_dir = sys.argv[1]
    clean_directory(target_dir)