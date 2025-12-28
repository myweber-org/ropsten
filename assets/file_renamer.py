
import os
import sys
from datetime import datetime

def rename_files_with_timestamp(directory):
    try:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory.")
            return False

        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        if not files:
            print("No files found in directory.")
            return True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        renamed_count = 0

        for filename in files:
            name, ext = os.path.splitext(filename)
            new_name = f"{timestamp}_{name}{ext}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
                renamed_count += 1
            except OSError as e:
                print(f"Failed to rename {filename}: {e}")

        print(f"Renamed {renamed_count} file(s).")
        return True

    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    success = rename_files_with_timestamp(target_directory)
    sys.exit(0 if success else 1)