
import os
import shutil

def organize_files(directory):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_extension = filename.split('.')[-1]
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            source_path = os.path.join(directory, filename)
            target_path = os.path.join(target_dir, filename)
            shutil.move(source_path, target_path)
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ")
    if os.path.exists(target_directory):
        organize_files(target_directory)
    else:
        print("Directory does not exist.")
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """Organize files in the given directory by their extensions."""
    base_path = Path(directory).resolve()
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    extension_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
        'Documents': ['.pdf', '.docx', '.txt', '.md', '.xlsx', '.pptx', '.csv'],
        'Archives': ['.zip', '.tar', '.gz', '.7z', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.json'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'Video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
    }
    
    other_folder = base_path / 'Other'
    
    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            for category, extensions in extension_categories.items():
                if file_extension in extensions:
                    target_folder = base_path / category
                    target_folder.mkdir(exist_ok=True)
                    
                    try:
                        shutil.move(str(item), str(target_folder / item.name))
                        print(f"Moved: {item.name} -> {category}/")
                    except Exception as e:
                        print(f"Failed to move {item.name}: {e}")
                    moved = True
                    break
            
            if not moved:
                other_folder.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(other_folder / item.name))
                    print(f"Moved: {item.name} -> Other/")
                except Exception as e:
                    print(f"Failed to move {item.name}: {e}")
    
    print("File organization completed.")

if __name__ == "__main__":
    target_dir = input("Enter directory path to organize (press Enter for current): ").strip()
    if not target_dir:
        target_dir = "."
    organize_files(target_dir)
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    named after their file extensions.
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
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
    print("File organization complete.")
import os
import shutil

def organize_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ")
    if os.path.exists(target_directory):
        organize_files(target_directory)
        print("File organization completed.")
    else:
        print("Directory does not exist.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
    subdirectories named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_ext = Path(item).suffix.lower()

            if file_ext:
                target_dir = os.path.join(directory, file_ext[1:] + "_files")
            else:
                target_dir = os.path.join(directory, "no_extension_files")

            os.makedirs(target_dir, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_dir, item))
                print(f"Moved: {item} -> {target_dir}")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)