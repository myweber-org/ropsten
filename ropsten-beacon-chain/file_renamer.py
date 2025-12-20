
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    try:
        files = os.listdir(directory)
        for filename in files:
            old_path = os.path.join(directory, filename)
            if os.path.isfile(old_path):
                new_filename = re.sub(pattern, replacement, filename)
                if new_filename != filename:
                    new_path = os.path.join(directory, new_filename)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    regex_pattern = sys.argv[2]
    replace_with = sys.argv[3]
    
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)
    
    rename_files(target_dir, regex_pattern, replace_with)