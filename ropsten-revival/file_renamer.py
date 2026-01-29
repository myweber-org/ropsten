
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
    
    rename_files_with_sequence(dir_path, prefix, extension)