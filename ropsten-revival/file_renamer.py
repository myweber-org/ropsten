import os
import sys
from datetime import datetime

def rename_files_by_date(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                stat = os.stat(filepath)
                ctime = stat.st_ctime
                date_str = datetime.fromtimestamp(ctime).strftime("%Y%m%d_%H%M%S")
                name, ext = os.path.splitext(filename)
                new_filename = f"{date_str}{ext}"
                new_filepath = os.path.join(directory, new_filename)
                os.rename(filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Failed to rename {filename}: {e}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory>")
        sys.exit(1)
    target_dir = sys.argv[1]
    rename_files_by_date(target_dir)
import os
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
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        sys.exit(1)
    
    rename_files_with_sequence(dir_path, prefix)
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
            return False
        
        files = os.listdir(directory)
        renamed_count = 0
        
        for filename in files:
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                new_name = re.sub(pattern, replacement, filename)
                
                if new_name != filename:
                    new_path = os.path.join(directory, new_name)
                    
                    if os.path.exists(new_path):
                        print(f"Warning: '{new_name}' already exists. Skipping '{filename}'.")
                        continue
                    
                    try:
                        os.rename(file_path, new_path)
                        print(f"Renamed: '{filename}' -> '{new_name}'")
                        renamed_count += 1
                    except OSError as e:
                        print(f"Error renaming '{filename}': {e}")
        
        print(f"\nRenaming complete. {renamed_count} files renamed.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./photos 'IMG_\\d+' 'Vacation_'")
        sys.exit(1)
    
    directory = sys.argv[1]
    pattern = sys.argv[2]
    replacement = sys.argv[3]
    
    rename_files(directory, pattern, replacement)

if __name__ == "__main__":
    main()import os
import sys

def rename_files(directory, prefix="file"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix]")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    rename_files(target_dir, name_prefix)