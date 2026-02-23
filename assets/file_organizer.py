
import os
import shutil
from pathlib import Path

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    extensions_folders = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv'],
        'archives': ['.zip', '.tar', '.gz', '.rar'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            file_ext = Path(item).suffix.lower()
            moved = False
            for folder, ext_list in extensions_folders.items():
                if file_ext in ext_list:
                    target_folder = os.path.join(directory, folder)
                    os.makedirs(target_folder, exist_ok=True)
                    shutil.move(item_path, os.path.join(target_folder, item))
                    moved = True
                    break
            if not moved:
                other_folder = os.path.join(directory, 'other')
                os.makedirs(other_folder, exist_ok=True)
                shutil.move(item_path, os.path.join(other_folder, item))

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization completed.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:] + "_files"

            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:] + "_files"
            
            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
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
            
            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            target_path = os.path.join(target_folder, item)
            
            try:
                shutil.move(item_path, target_path)
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
    subdirectories based on their file extensions.
    """
    # Define file type categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    # Ensure the directory exists
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Create category directories if they don't exist
    for category in categories.keys():
        category_dir = target_dir / category
        category_dir.mkdir(exist_ok=True)

    # Track moved files and errors
    moved_files = []
    errors = []

    # Iterate over all items in the directory
    for item in target_dir.iterdir():
        # Skip directories and hidden files
        if item.is_dir() or item.name.startswith('.'):
            continue

        # Get file extension
        file_extension = item.suffix.lower()

        # Find the appropriate category
        target_category = None
        for category, extensions in categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # If no category found, place in 'Other'
        if not target_category:
            target_category = 'Other'
            other_dir = target_dir / target_category
            other_dir.mkdir(exist_ok=True)

        # Define target path
        target_path = target_dir / target_category / item.name

        # Handle file name conflicts
        counter = 1
        original_stem = item.stem
        while target_path.exists():
            new_name = f"{original_stem}_{counter}{item.suffix}"
            target_path = target_dir / target_category / new_name
            counter += 1

        # Move the file
        try:
            shutil.move(str(item), str(target_path))
            moved_files.append((item.name, target_category))
        except Exception as e:
            errors.append((item.name, str(e)))

    # Print summary
    if moved_files:
        print(f"Successfully organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

    if errors:
        print(f"\nEncountered {len(errors)} error(s):")
        for filename, error_msg in errors:
            print(f"  {filename}: {error_msg}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subdirectories
    based on their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                target_dir = os.path.join(directory, file_extension[1:])
            else:
                target_dir = os.path.join(directory, "no_extension")
            
            os.makedirs(target_dir, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_dir, item))
                print(f"Moved: {item} -> {target_dir}")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    }
    
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Get all files in the directory
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return
    
    moved_count = 0
    
    for filename in files:
        file_extension = os.path.splitext(filename)[1].lower()
        target_category = None
        
        # Find the category for the file extension
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break
        
        # If no category found, use 'Other'
        if target_category is None:
            target_category = 'Other'
        
        # Create target directory if it doesn't exist
        target_dir = os.path.join(directory, target_category)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Move the file
        source_path = os.path.join(directory, filename)
        target_path = os.path.join(target_dir, filename)
        
        try:
            shutil.move(source_path, target_path)
            moved_count += 1
            print(f"Moved: {filename} -> {target_category}/")
        except Exception as e:
            print(f"Failed to move {filename}: {e}")
    
    print(f"\nOrganization complete. Moved {moved_count} file(s).")

if __name__ == "__main__":
    # Use current directory if no argument provided
    import sys
    target_directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_folder = os.path.join(directory, file_extension.upper() + "_FILES")
            os.makedirs(target_folder, exist_ok=True)
            shutil.move(file_path, os.path.join(target_folder, filename))
            print(f"Moved {filename} to {target_folder}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ")
    if os.path.isdir(target_directory):
        organize_files(target_directory)
        print("File organization completed.")
    else:
        print("Invalid directory path.")
import os
import shutil

def organize_files(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1].lower() if '.' in filename else 'no_extension'
            target_folder = os.path.join(directory_path, file_extension)

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            try:
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f"Moved: {filename} -> {file_extension}/")
            except Exception as e:
                print(f"Failed to move {filename}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
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
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:]

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