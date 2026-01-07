
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