
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
        print(f"Error: {e}")