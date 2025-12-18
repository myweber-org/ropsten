
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into subfolders
    named after their file extensions.
    """
    # Convert the input path to a Path object for easier handling
    path = Path(directory_path)

    # Check if the provided path exists and is a directory
    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    # Dictionary to map file extensions to folder names
    extension_categories = {
        '.txt': 'TextFiles',
        '.pdf': 'PDFs',
        '.jpg': 'Images',
        '.jpeg': 'Images',
        '.png': 'Images',
        '.mp3': 'Audio',
        '.mp4': 'Videos',
        '.zip': 'Archives',
        '.py': 'PythonScripts',
    }

    # Iterate over all items in the directory
    for item in path.iterdir():
        # Skip if it's a directory
        if item.is_dir():
            continue

        # Get the file extension (lowercase for consistency)
        file_extension = item.suffix.lower()

        # Determine the target folder name
        # Use the category if defined, otherwise use 'Other'
        folder_name = extension_categories.get(file_extension, 'Other')
        target_folder = path / folder_name

        # Create the target folder if it doesn't exist
        target_folder.mkdir(exist_ok=True)

        # Construct the target file path
        target_file_path = target_folder / item.name

        # Check if a file with the same name already exists in the target folder
        if target_file_path.exists():
            print(f"Warning: '{item.name}' already exists in '{folder_name}'. Skipping.")
            continue

        # Move the file to the target folder
        try:
            shutil.move(str(item), str(target_folder))
            print(f"Moved: {item.name} -> {folder_name}/")
        except Exception as e:
            print(f"Error moving {item.name}: {e}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    target_directory = input("Enter the directory path to organize (or press Enter for current directory): ").strip()
    if not target_directory:
        target_directory = os.getcwd()
    organize_files(target_directory)