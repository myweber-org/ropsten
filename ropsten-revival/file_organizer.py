
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the given directory by moving them into subfolders
    named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    base_path = Path(directory_path)

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if not file_extension:
                file_extension = "no_extension"

            target_folder_name = file_extension[1:] if file_extension.startswith('.') else file_extension
            target_folder = base_path / target_folder_name

            target_folder.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {target_folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the specified directory by moving them into
    subdirectories named after their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = os.path.join(directory_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into subfolders
    named after their file extensions.
    """
    # Convert to Path object for easier handling
    base_path = Path(directory_path)

    # Check if the directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist or is not a directory.")
        return

    # Iterate over all items in the directory
    for item in base_path.iterdir():
        # Skip if it's a directory
        if item.is_dir():
            continue

        # Get the file extension (without the dot)
        extension = item.suffix[1:].lower() if item.suffix else 'no_extension'

        # Create a folder name based on the extension
        folder_name = extension if extension else 'no_extension'
        target_folder = base_path / folder_name

        # Create the target folder if it doesn't exist
        target_folder.mkdir(exist_ok=True)

        # Construct the target file path
        target_file_path = target_folder / item.name

        # Check if a file with the same name already exists in the target folder
        counter = 1
        while target_file_path.exists():
            # Append a number to the filename to avoid overwriting
            new_name = f"{item.stem}_{counter}{item.suffix}"
            target_file_path = target_folder / new_name
            counter += 1

        # Move the file
        try:
            shutil.move(str(item), str(target_file_path))
            print(f"Moved: {item.name} -> {target_folder.name}/")
        except Exception as e:
            print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)
    print("File organization complete.")