
import os
import sys
from datetime import datetime

def rename_files_by_date(directory):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            creation_time = os.path.getctime(filepath)
            date_str = datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
            
            name, extension = os.path.splitext(filename)
            new_filename = f"{date_str}{extension}"
            new_filepath = os.path.join(directory, new_filename)
            
            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{date_str}_{counter}{extension}"
                new_filepath = os.path.join(directory, new_filename)
                counter += 1
            
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    if not os.path.isdir(target_directory):
        print(f"Error: {target_directory} is not a valid directory")
        sys.exit(1)
    
    rename_files_by_date(target_directory)