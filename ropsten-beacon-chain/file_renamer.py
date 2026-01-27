
import os
import sys
from pathlib import Path

def rename_files_sequentially(directory_path, prefix="file"):
    try:
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            print(f"Error: {directory_path} is not a valid directory.")
            return False

        files = []
        for item in path.iterdir():
            if item.is_file():
                mtime = item.stat().st_mtime
                files.append((mtime, item))

        files.sort(key=lambda x: x[0])

        for index, (_, file_path) in enumerate(files, start=1):
            extension = file_path.suffix
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = file_path.parent / new_name

            try:
                file_path.rename(new_path)
                print(f"Renamed: {file_path.name} -> {new_name}")
            except Exception as e:
                print(f"Failed to rename {file_path.name}: {e}")

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)

    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    rename_files_sequentially(dir_path, prefix)
import os
import datetime
import sys

def add_timestamp_prefix(filepath):
    """Add a timestamp prefix to the filename."""
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' does not exist."
    
    directory, filename = os.path.split(filepath)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}_{filename}"
    new_filepath = os.path.join(directory, new_filename)
    
    try:
        os.rename(filepath, new_filepath)
        return f"Renamed '{filename}' to '{new_filename}'"
    except Exception as e:
        return f"Error renaming file: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <filepath1> [filepath2 ...]")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        result = add_timestamp_prefix(filepath)
        print(result)

if __name__ == "__main__":
    main()
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
        
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix] [extension]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_with_sequence(dir_path, prefix, extension)