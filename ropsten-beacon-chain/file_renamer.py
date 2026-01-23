
import os
import sys
from datetime import datetime

def rename_files_by_date(directory, prefix="file"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            creation_time = os.path.getctime(filepath)
            date_str = datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
            
            name, ext = os.path.splitext(filename)
            new_filename = f"{prefix}_{date_str}{ext}"
            new_filepath = os.path.join(directory, new_filename)
            
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix]")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    file_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory")
        sys.exit(1)
    
    success = rename_files_by_date(target_dir, file_prefix)
    sys.exit(0 if success else 1)