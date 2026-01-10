
import os
import sys

def rename_files(directory, prefix="file_"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}{index:03d}{extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file_"
    
    rename_files(target_directory, name_prefix)
import os
import sys
import argparse

def rename_files_with_sequence(directory, prefix, start_number=1, extension=None):
    """
    Rename files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to directory containing files
        prefix (str): Prefix for renamed files
        start_number (int): Starting number for sequence
        extension (str, optional): Filter files by extension
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return False
    
    try:
        files = os.listdir(directory)
        
        if extension:
            files = [f for f in files if f.lower().endswith(f'.{extension.lower()}')]
        
        files.sort()
        
        for index, filename in enumerate(files, start=start_number):
            old_path = os.path.join(directory, filename)
            
            if extension:
                new_filename = f"{prefix}_{index:03d}.{extension}"
            else:
                file_ext = os.path.splitext(filename)[1]
                new_filename = f"{prefix}_{index:03d}{file_ext}"
            
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")
        
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Rename files with sequential numbering')
    parser.add_argument('directory', help='Directory containing files to rename')
    parser.add_argument('prefix', help='Prefix for renamed files')
    parser.add_argument('--start', type=int, default=1, help='Starting number (default: 1)')
    parser.add_argument('--ext', help='Filter by file extension (e.g., jpg, png)')
    
    args = parser.parse_args()
    
    success = rename_files_with_sequence(
        args.directory,
        args.prefix,
        args.start,
        args.ext
    )
    
    if success:
        print("File renaming completed successfully.")
    else:
        print("File renaming failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()