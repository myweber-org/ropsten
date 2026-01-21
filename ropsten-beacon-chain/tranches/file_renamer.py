
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        prefix (str): Prefix for the new filenames.
        extension (str): File extension to filter and use for renaming.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]
    files.sort()
    
    if not files:
        print(f"No files with extension '{extension}' found in '{directory}'.")
        return
    
    print(f"Renaming {len(files)} files in '{directory}':")
    
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_name = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_name)
        
        try:
            os.rename(old_path, new_path)
            print(f"  '{filename}' -> '{new_name}'")
        except OSError as e:
            print(f"  Error renaming '{filename}': {e}")
    
    print("Renaming completed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        print("Example: python file_renamer.py ./documents image .jpg")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix_arg = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension_arg = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_with_sequence(dir_path, prefix_arg, extension_arg)