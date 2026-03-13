
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): String to replace matched pattern with.
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False

        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        renamed_count = 0

        for filename in files:
            new_name = re.sub(pattern, replacement, filename)
            if new_name != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                    renamed_count += 1
                except OSError as e:
                    print(f"Failed to rename {filename}: {e}")

        print(f"Renaming complete. {renamed_count} files renamed.")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        sys.exit(1)

    dir_path = sys.argv[1]
    regex_pattern = sys.argv[2]
    replace_with = sys.argv[3]

    success = rename_files(dir_path, regex_pattern, replace_with)
    sys.exit(0 if success else 1)
import os
import sys

def rename_files(directory, prefix="file_"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            file_extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}{index:03d}{file_extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    custom_prefix = sys.argv[2] if len(sys.argv) > 2 else "file_"
    
    rename_files(dir_path, custom_prefix)