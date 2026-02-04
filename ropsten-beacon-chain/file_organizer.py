
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ")
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv']
    }

    # Ensure the directory exists
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in categories:
        category_path = dir_path / category
        category_path.mkdir(exist_ok=True)

    # Track moved files and unknown extensions
    moved_files = []
    unknown_extensions = set()

    # Iterate over all items in the directory
    for item in dir_path.iterdir():
        # Skip directories and hidden files
        if item.is_dir() or item.name.startswith('.'):
            continue

        # Get the file extension
        extension = item.suffix.lower()

        # Find the appropriate category for the file
        target_category = None
        for category, extensions in categories.items():
            if extension in extensions:
                target_category = category
                break

        # Move the file to the corresponding category folder
        if target_category:
            target_path = dir_path / target_category / item.name
            # Handle duplicate filenames
            counter = 1
            while target_path.exists():
                name_parts = item.stem, item.suffix
                new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                target_path = dir_path / target_category / new_name
                counter += 1

            try:
                shutil.move(str(item), str(target_path))
                moved_files.append((item.name, target_category))
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")
        else:
            unknown_extensions.add(extension)

    # Print summary
    if moved_files:
        print(f"Successfully organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were organized.")

    if unknown_extensions:
        print(f"\nFiles with unknown extensions were not moved: {', '.join(sorted(unknown_extensions))}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_dir = os.getcwd()
    organize_files(current_dir)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            folder_path = os.path.join(directory, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            destination = os.path.join(folder_path, item)
            shutil.move(item_path, destination)
            print(f"Moved: {item} -> {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)