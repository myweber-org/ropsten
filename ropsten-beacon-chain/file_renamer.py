
import os
import glob
from pathlib import Path
from datetime import datetime

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    files = glob.glob(os.path.join(directory, f"*{extension}"))
    
    files_with_mtime = []
    for file_path in files:
        mtime = os.path.getmtime(file_path)
        files_with_mtime.append((mtime, file_path))
    
    files_with_mtime.sort(key=lambda x: x[0])
    
    for index, (_, file_path) in enumerate(files_with_mtime, start=1):
        new_filename = f"{prefix}_{index:03d}{extension}"
        new_filepath = os.path.join(directory, new_filename)
        
        try:
            os.rename(file_path, new_filepath)
            print(f"Renamed: {Path(file_path).name} -> {new_filename}")
        except OSError as e:
            print(f"Error renaming {file_path}: {e}")

if __name__ == "__main__":
    target_directory = "./documents"
    
    if not os.path.exists(target_directory):
        print(f"Directory '{target_directory}' does not exist.")
    else:
        rename_files_sequentially(target_directory, prefix="document", extension=".pdf")
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the specified directory with sequential numbering.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()
    
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_filename = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        except OSError as e:
            print(f"Failed to rename {filename}: {e}")
            return False
    
    print(f"Successfully renamed {len(files)} files.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix_arg = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension_arg = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_with_sequence(dir_path, prefix_arg, extension_arg)