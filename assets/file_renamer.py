
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): Replacement string for matched pattern.
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return

        files = os.listdir(directory)
        renamed_count = 0

        for filename in files:
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                new_filename = re.sub(pattern, replacement, filename)
                
                if new_filename != filename:
                    new_file_path = os.path.join(directory, new_filename)
                    
                    if os.path.exists(new_file_path):
                        print(f"Warning: '{new_filename}' already exists. Skipping '{filename}'.")
                        continue
                    
                    os.rename(file_path, new_file_path)
                    print(f"Renamed: '{filename}' -> '{new_filename}'")
                    renamed_count += 1

        print(f"\nRenaming complete. {renamed_count} file(s) renamed.")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./files '\\d+' 'NUM'")
        sys.exit(1)

    directory = sys.argv[1]
    pattern = sys.argv[2]
    replacement = sys.argv[3]

    rename_files(directory, pattern, replacement)

if __name__ == "__main__":
    main()