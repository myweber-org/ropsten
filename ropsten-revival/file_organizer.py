
import os
import shutil

def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()

            if extension:
                target_folder = os.path.join(directory, extension[1:] + "_files")
            else:
                target_folder = os.path.join(directory, "no_extension_files")

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            try:
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f"Moved: {filename} -> {target_folder}")
            except Exception as e:
                print(f"Failed to move {filename}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
    subfolders named after their file extensions.
    """
    base_path = Path(directory)
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    for item in base_path.iterdir():
        if item.is_file():
            extension = item.suffix.lower()
            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = base_path / folder_name
            target_folder.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")