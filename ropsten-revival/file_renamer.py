
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
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
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix] [extension]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        sys.exit(1)
    
    rename_files_with_sequence(dir_path, prefix, extension)import os
import sys

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    """
    Renames all files in the specified directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        prefix (str): Prefix for the new filenames.
        extension (str): File extension to filter and apply.
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False
        
        files = [f for f in os.listdir(directory) 
                 if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]
        files.sort()
        
        if not files:
            print(f"No files with extension '{extension}' found in '{directory}'.")
            return True
        
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = os.path.join(directory, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
            except OSError as e:
                print(f"Failed to rename {filename}: {e}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        print("Example: python file_renamer.py ./documents document .pdf")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix_arg = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension_arg = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    success = rename_files_sequentially(dir_path, prefix_arg, extension_arg)
    sys.exit(0 if success else 1)import os
import re
import argparse

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): String to replace matched pattern with.
    """
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return

    renamed_count = 0
    for filename in files:
        filepath = os.path.join(directory, filename)
        
        if not os.path.isfile(filepath):
            continue

        new_filename = re.sub(pattern, replacement, filename)
        
        if new_filename != filename:
            new_filepath = os.path.join(directory, new_filename)
            
            if os.path.exists(new_filepath):
                print(f"Warning: '{new_filename}' already exists. Skipping '{filename}'.")
                continue
            
            try:
                os.rename(filepath, new_filepath)
                print(f"Renamed: '{filename}' -> '{new_filename}'")
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming '{filename}': {e}")

    print(f"\nRenaming complete. {renamed_count} files renamed.")

def main():
    parser = argparse.ArgumentParser(description="Rename files in a directory using regex patterns.")
    parser.add_argument("directory", help="Directory containing files to rename")
    parser.add_argument("pattern", help="Regex pattern to match in filenames")
    parser.add_argument("replacement", help="Replacement string for matched pattern")
    
    args = parser.parse_args()
    
    rename_files(args.directory, args.pattern, args.replacement)

if __name__ == "__main__":
    main()
import os
import sys
from pathlib import Path
from datetime import datetime

def rename_files_sequentially(directory, prefix="file"):
    try:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            print(f"Error: Directory '{directory}' does not exist.")
            return False

        files = []
        for item in path.iterdir():
            if item.is_file():
                try:
                    creation_time = item.stat().st_ctime
                except OSError:
                    creation_time = 0
                files.append((creation_time, item))

        files.sort(key=lambda x: x[0])

        for index, (_, file_path) in enumerate(files, start=1):
            extension = file_path.suffix
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = file_path.parent / new_name

            try:
                file_path.rename(new_path)
                print(f"Renamed: {file_path.name} -> {new_name}")
            except OSError as e:
                print(f"Failed to rename {file_path.name}: {e}")

        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
        rename_files_sequentially(target_dir, prefix)
    else:
        print("Usage: python file_renamer.py <directory> [prefix]")
import os
import glob
from pathlib import Path

def rename_files_sequential(directory, prefix="file", extension=".txt"):
    files = list(Path(directory).glob(f"*{extension}"))
    files.sort(key=lambda x: x.stat().st_ctime)
    
    for index, file_path in enumerate(files, start=1):
        new_name = f"{prefix}_{index:03d}{extension}"
        new_path = file_path.parent / new_name
        file_path.rename(new_path)
        print(f"Renamed: {file_path.name} -> {new_name}")

if __name__ == "__main__":
    target_dir = input("Enter directory path: ").strip()
    if os.path.isdir(target_dir):
        rename_files_sequential(target_dir)
    else:
        print("Invalid directory path")
import os
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
    
    dir_path = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    rename_files(dir_path, name_prefix)
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the given directory with sequential numbering.
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
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix] [extension]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        sys.exit(1)
    
    rename_files_with_sequence(dir_path, prefix, extension)