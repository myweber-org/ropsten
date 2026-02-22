
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
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        remove_duplicates(input_file, output_file)
        print(f"Duplicates removed. Cleaned file saved as {output_file}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")import os
import glob

def clean_temp_files(directory, patterns):
    """
    Remove temporary files matching given patterns from a directory.
    
    Args:
        directory (str): Path to the directory to clean.
        patterns (list): List of file patterns to match (e.g., ['*.tmp', 'temp_*']).
    
    Returns:
        int: Number of files removed.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    removed_count = 0
    for pattern in patterns:
        search_path = os.path.join(directory, pattern)
        for file_path in glob.glob(search_path):
            try:
                os.remove(file_path)
                removed_count += 1
                print(f"Removed: {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}")
    
    return removed_count

if __name__ == "__main__":
    target_dir = "/tmp/test_cleanup"
    temp_patterns = ["*.tmp", "temp_*", "*.bak"]
    
    try:
        count = clean_temp_files(target_dir, temp_patterns)
        print(f"Cleaning completed. Removed {count} files.")
    except ValueError as e:
        print(f"Error: {e}")import sys
import hashlib

def remove_duplicates(input_file, output_file=None):
    seen_lines = set()
    unique_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_hash = hashlib.md5(line.strip().encode()).hexdigest()
                if line_hash not in seen_lines:
                    seen_lines.add(line_hash)
                    unique_lines.append(line)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    
    if not output_file:
        output_file = input_file + '.deduped'
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        print(f"Successfully removed duplicates. Output saved to '{output_file}'")
        return True
    except IOError as e:
        print(f"Error writing to file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)