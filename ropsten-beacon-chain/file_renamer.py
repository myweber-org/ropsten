
import os
import sys

def rename_files_with_sequential_numbers(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the specified directory with sequential numbering.
    """
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            new_filename = f"{prefix}_{index:03d}{extension}"
            new_path = os.path.join(directory, new_filename)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        
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
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix] [extension]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_with_sequential_numbers(dir_path, prefix, extension)
import os
import glob
from pathlib import Path
from datetime import datetime

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    files = glob.glob(os.path.join(directory, "*" + extension))
    files.sort(key=os.path.getctime)
    
    for index, file_path in enumerate(files, start=1):
        new_name = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_name)
        os.rename(file_path, new_path)
        print(f"Renamed: {Path(file_path).name} -> {new_name}")

if __name__ == "__main__":
    target_dir = "./documents"
    if os.path.exists(target_dir):
        rename_files_sequentially(target_dir, prefix="doc", extension=".pdf")
    else:
        print(f"Directory {target_dir} does not exist.")