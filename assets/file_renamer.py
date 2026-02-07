
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
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): String to replace matched pattern.
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return
        
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        renamed_count = 0
        
        for filename in files:
            new_name = re.sub(pattern, replacement, filename)
            if new_name != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)
                
                # Avoid overwriting existing files
                if os.path.exists(new_path):
                    print(f"Warning: '{new_name}' already exists. Skipping '{filename}'.")
                    continue
                
                os.rename(old_path, new_path)
                print(f"Renamed: '{filename}' -> '{new_name}'")
                renamed_count += 1
        
        print(f"\nRenaming complete. {renamed_count} file(s) renamed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./photos 'IMG_\\d+' 'Vacation'")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    regex_pattern = sys.argv[2]
    replace_with = sys.argv[3]
    
    rename_files(target_dir, regex_pattern, replace_with)import os
import sys

def rename_files_with_sequence(directory, prefix="file"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            file_extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{file_extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
    
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return False
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        rename_files_with_sequence(target_dir)
    else:
        print("Usage: python file_renamer.py <directory_path>")
import os
import glob
from pathlib import Path
from datetime import datetime

def rename_files_with_timestamp(directory, prefix="file", extension=".txt"):
    files = sorted(glob.glob(os.path.join(directory, "*" + extension)), key=os.path.getctime)
    
    for index, file_path in enumerate(files, start=1):
        creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
        timestamp_str = creation_time.strftime("%Y%m%d_%H%M%S")
        new_name = f"{prefix}_{timestamp_str}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_name)
        
        try:
            os.rename(file_path, new_path)
            print(f"Renamed: {Path(file_path).name} -> {new_name}")
        except OSError as e:
            print(f"Error renaming {file_path}: {e}")

if __name__ == "__main__":
    target_directory = "./documents"
    if os.path.exists(target_directory):
        rename_files_with_timestamp(target_directory, prefix="document", extension=".pdf")
    else:
        print(f"Directory {target_directory} does not exist.")