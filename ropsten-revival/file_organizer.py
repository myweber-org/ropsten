
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
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
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
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization completed.")
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory by their extensions.
    Creates subdirectories for each file type and moves files accordingly.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension[1:].lower() if extension else 'no_extension'

            target_dir = os.path.join(directory, extension)
            os.makedirs(target_dir, exist_ok=True)

            target_path = os.path.join(target_dir, filename)
            shutil.move(file_path, target_path)
            print(f"Moved: {filename} -> {extension}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
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
            
            target_path = os.path.join(target_folder, item)
            shutil.move(item_path, target_path)
            print(f"Moved: {item} -> {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
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
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    # Ensure the directory exists
    target_dir = Path(directory)
    if not target_dir.exists():
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = target_dir / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files and errors
    moved_files = []
    errors = []
    
    # Iterate through all items in the directory
    for item in target_dir.iterdir():
        # Skip directories and hidden files
        if item.is_dir() or item.name.startswith('.'):
            continue
        
        # Get file extension
        file_ext = item.suffix.lower()
        
        # Find the appropriate category for the file
        moved = False
        for category, extensions in file_categories.items():
            if file_ext in extensions:
                destination = target_dir / category / item.name
                try:
                    # Handle naming conflicts
                    if destination.exists():
                        base_name = item.stem
                        counter = 1
                        while destination.exists():
                            new_name = f"{base_name}_{counter}{item.suffix}"
                            destination = target_dir / category / new_name
                            counter += 1
                    
                    shutil.move(str(item), str(destination))
                    moved_files.append((item.name, category))
                    moved = True
                    break
                except Exception as e:
                    errors.append((item.name, str(e)))
        
        # If file doesn't match any category, move to 'Other'
        if not moved:
            other_dir = target_dir / 'Other'
            other_dir.mkdir(exist_ok=True)
            destination = other_dir / item.name
            try:
                shutil.move(str(item), str(destination))
                moved_files.append((item.name, 'Other'))
            except Exception as e:
                errors.append((item.name, str(e)))
    
    # Print summary
    print(f"Organization complete for '{directory}'")
    print(f"Total files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for filename, error_msg in errors:
            print(f"  {filename}: {error_msg}")

if __name__ == "__main__":
    # Use current directory if no argument provided
    import sys
    target_directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    organize_files(target_directory)