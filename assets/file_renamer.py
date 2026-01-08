
import os
import re
import sys
from pathlib import Path

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory by replacing parts of the filename
    matching the regex pattern with the replacement string.
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Error: '{directory}' is not a valid directory.")
            return False

        files_renamed = 0
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                old_name = file_path.name
                new_name = re.sub(pattern, replacement, old_name)
                if new_name != old_name:
                    new_path = file_path.with_name(new_name)
                    try:
                        file_path.rename(new_path)
                        print(f"Renamed: '{old_name}' -> '{new_name}'")
                        files_renamed += 1
                    except OSError as e:
                        print(f"Error renaming '{old_name}': {e}")

        print(f"Total files renamed: {files_renamed}")
        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        sys.exit(1)

    target_dir = sys.argv[1]
    regex_pattern = sys.argv[2]
    repl_string = sys.argv[3]

    success = rename_files(target_dir, regex_pattern, repl_string)
    sys.exit(0 if success else 1)
import os
import sys
from pathlib import Path
from datetime import datetime

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

        for index, (mtime, file_path) in enumerate(files, start=1):
            extension = file_path.suffix
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = file_path.parent / new_name

            if new_path.exists():
                print(f"Warning: {new_name} already exists. Skipping rename.")
                continue

            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")

        print("Renaming completed successfully.")
        return True

    except PermissionError:
        print("Error: Permission denied. Check file access rights.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)

    dir_path = sys.argv[1]
    pref = sys.argv[2] if len(sys.argv) > 2 else "file"

    rename_files_sequentially(dir_path, pref)