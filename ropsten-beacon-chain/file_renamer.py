
import os
import sys
import argparse

def rename_files_with_sequence(directory, prefix="file", extension=".txt", start_number=1):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename
        prefix (str): Prefix for renamed files
        extension (str): File extension to filter and apply
        start_number (int): Starting number for the sequence
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    if not files:
        print("No files found in the directory.")
        return
    
    counter = start_number
    for filename in files:
        old_path = os.path.join(directory, filename)
        name, ext = os.path.splitext(filename)
        
        if extension and not filename.endswith(extension):
            continue
        
        new_filename = f"{prefix}_{counter:03d}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            counter += 1
        except OSError as e:
            print(f"Error renaming {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Rename files with sequential numbering.")
    parser.add_argument("directory", help="Directory containing files to rename")
    parser.add_argument("-p", "--prefix", default="file", help="Prefix for renamed files")
    parser.add_argument("-e", "--extension", default=".txt", help="File extension filter")
    parser.add_argument("-s", "--start", type=int, default=1, help="Starting number for sequence")
    
    args = parser.parse_args()
    
    rename_files_with_sequence(
        directory=args.directory,
        prefix=args.prefix,
        extension=args.extension,
        start_number=args.start
    )

if __name__ == "__main__":
    main()