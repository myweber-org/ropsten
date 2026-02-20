
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
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:].upper() + "_FILES"
            else:
                folder_name = "NO_EXTENSION_FILES"
            
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
    print("File organization completed.")
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by their extensions.
    Creates subdirectories for each file type and moves files accordingly.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()

            if extension:
                target_dir = os.path.join(directory, extension[1:])
            else:
                target_dir = os.path.join(directory, "no_extension")

            os.makedirs(target_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved '{filename}' to '{target_dir}'")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    base_path = Path(directory)
    
    # Define categories and their associated file extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mov", ".mkv"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"],
        "Executables": [".exe", ".msi", ".sh", ".bat"],
    }
    
    # Create category folders if they don't exist
    for category in categories:
        (base_path / category).mkdir(exist_ok=True)
    
    # Create an 'Other' folder for uncategorized files
    other_folder = base_path / "Other"
    other_folder.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    
    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category
            for category, extensions in categories.items():
                if file_extension in extensions:
                    target_folder = base_path / category
                    try:
                        shutil.move(str(item), str(target_folder / item.name))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            # If file doesn't match any category, move to 'Other'
            if not moved:
                try:
                    shutil.move(str(item), str(other_folder / item.name))
                    moved_files.append((item.name, "Other"))
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
    
    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

if __name__ == "__main__":
    organize_files()
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subfolders
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
                folder_name = file_extension[1:].upper() + "_Files"
            else:
                folder_name = "NO_EXTENSION_Files"
            
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
    print("File organization completed.")
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()

            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory, folder_name)

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            try:
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f"Moved: {filename} -> {folder_name}/")
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
    
    # Convert directory to Path object for easier handling
    base_path = Path(directory)
    
    # Ensure the directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return
    
    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = base_path / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files and errors
    moved_files = []
    error_files = []
    
    # Iterate through all files in the directory
    for item in base_path.iterdir():
        # Skip directories and hidden files
        if item.is_dir() or item.name.startswith('.'):
            continue
        
        # Get file extension
        file_extension = item.suffix.lower()
        
        # Find the appropriate category for the file
        target_category = None
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break
        
        # If no category found, move to 'Other'
        if target_category is None:
            target_category = 'Other'
            other_path = base_path / target_category
            other_path.mkdir(exist_ok=True)
        
        # Construct target path
        target_path = base_path / target_category / item.name
        
        # Check if file already exists in target location
        if target_path.exists():
            # Add a counter to create a unique filename
            counter = 1
            name_parts = item.stem, item.suffix
            while target_path.exists():
                new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                target_path = base_path / target_category / new_name
                counter += 1
        
        # Move the file
        try:
            shutil.move(str(item), str(target_path))
            moved_files.append((item.name, target_category))
        except Exception as e:
            error_files.append((item.name, str(e)))
    
    # Print summary
    print(f"Organization complete for directory: {directory}")
    print(f"Files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if error_files:
        print(f"\nErrors ({len(error_files)} files):")
        for filename, error in error_files:
            print(f"  {filename}: {error}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_dir = os.getcwd()
    organize_files(current_dir)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_ext = Path(item).suffix.lower()

            if file_ext:
                folder_name = file_ext[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, target_folder)
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter the directory path to organize: ").strip()
    organize_files(target_dir)
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
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return
    
    for item in base_path.iterdir():
        if item.is_file():
            extension = item.suffix.lower()
            if not extension:
                extension = "no_extension"
            else:
                extension = extension[1:]  # Remove the leading dot
            
            target_folder = base_path / extension
            target_folder.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {extension}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

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
        filepath = os.path.join(directory, filename)

        if os.path.isfile(filepath):
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            if ext:
                target_dir = os.path.join(directory, ext[1:] + "_files")
            else:
                target_dir = os.path.join(directory, "no_extension_files")

            os.makedirs(target_dir, exist_ok=True)
            shutil.move(filepath, os.path.join(target_dir, filename))
            print(f"Moved: {filename} -> {target_dir}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()

            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory, folder_name)

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            try:
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f"Moved: {filename} -> {folder_name}/")
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
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = target_dir / category
        category_path.mkdir(exist_ok=True)

    # Track moved files and errors
    moved_files = []
    error_files = []

    # Iterate over all items in the directory
    for item in target_dir.iterdir():
        # Skip directories and hidden files
        if item.is_dir() or item.name.startswith('.'):
            continue

        # Get file extension
        file_ext = item.suffix.lower()

        # Find the appropriate category for the file
        target_category = None
        for category, extensions in file_categories.items():
            if file_ext in extensions:
                target_category = category
                break

        # If no category found, move to 'Other'
        if target_category is None:
            target_category = 'Other'
            other_dir = target_dir / target_category
            other_dir.mkdir(exist_ok=True)

        # Define destination path
        dest_dir = target_dir / target_category
        dest_path = dest_dir / item.name

        # Handle file name conflicts
        counter = 1
        while dest_path.exists():
            stem = item.stem
            new_name = f"{stem}_{counter}{item.suffix}"
            dest_path = dest_dir / new_name
            counter += 1

        # Move the file
        try:
            shutil.move(str(item), str(dest_path))
            moved_files.append((item.name, target_category))
        except Exception as e:
            error_files.append((item.name, str(e)))

    # Print summary
    print(f"Organization complete for '{directory}'")
    print(f"Moved {len(moved_files)} files:")
    for filename, category in moved_files:
        print(f"  {filename} -> {category}/")

    if error_files:
        print(f"Encountered {len(error_files)} errors:")
        for filename, error in error_files:
            print(f"  {filename}: {error}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)