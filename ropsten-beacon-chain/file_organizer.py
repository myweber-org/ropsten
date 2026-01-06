
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
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
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
    }

    # Ensure the directory exists
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in categories:
        (dir_path / category).mkdir(exist_ok=True)

    # Track moved files and unknown extensions
    moved_files = []
    unknown_extensions = set()

    # Iterate over files in the directory
    for item in dir_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False

            # Find the appropriate category
            for category, extensions in categories.items():
                if file_ext in extensions:
                    target_dir = dir_path / category
                    try:
                        shutil.move(str(item), str(target_dir / item.name))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Failed to move {item.name}: {e}")

            # If no category matched, note the unknown extension
            if not moved:
                unknown_extensions.add(file_ext)

    # Print summary
    if moved_files:
        print("Files organized successfully:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

    if unknown_extensions:
        print(f"\nUnknown file extensions encountered: {', '.join(unknown_extensions)}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subdirectories based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            if ext:
                target_dir = os.path.join(directory, ext[1:] + "_files")
            else:
                target_dir = os.path.join(directory, "no_extension_files")

            os.makedirs(target_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved: {filename} -> {target_dir}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)