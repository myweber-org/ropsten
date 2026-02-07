
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