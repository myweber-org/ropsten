
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into
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
                folder_name = extension[1:] if extension.startswith('.') else extension
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
import os
import shutil
from pathlib import Path

def organize_files(source_dir, target_dir=None):
    """
    Organize files in source directory by their extensions.
    Creates subdirectories for each file type.
    """
    if target_dir is None:
        target_dir = source_dir
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist.")
        return
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    extension_categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
        'spreadsheets': ['.xls', '.xlsx', '.csv'],
        'presentations': ['.ppt', '.pptx'],
        'archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    }
    
    category_folders = {}
    for category in extension_categories.keys():
        category_path = target_path / category
        category_path.mkdir(exist_ok=True)
        category_folders[category] = category_path
    
    other_folder = target_path / 'other'
    other_folder.mkdir(exist_ok=True)
    
    files_organized = 0
    files_skipped = 0
    
    for item in source_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            for category, extensions in extension_categories.items():
                if file_extension in extensions:
                    destination = category_folders[category] / item.name
                    
                    if not destination.exists():
                        shutil.move(str(item), str(destination))
                        files_organized += 1
                        moved = True
                        break
                    else:
                        print(f"File {item.name} already exists in {category} folder.")
                        files_skipped += 1
                        moved = True
                        break
            
            if not moved:
                destination = other_folder / item.name
                if not destination.exists():
                    shutil.move(str(item), str(destination))
                    files_organized += 1
                else:
                    print(f"File {item.name} already exists in other folder.")
                    files_skipped += 1
    
    print(f"Organization complete. {files_organized} files organized, {files_skipped} files skipped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python file_organizer.py <source_directory> [target_directory]")
        sys.exit(1)
    
    source_directory = sys.argv[1]
    target_directory = sys.argv[2] if len(sys.argv) > 2 else None
    
    organize_files(source_directory, target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    base_path = Path(directory).resolve()
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:]  # Remove the leading dot

            target_folder = base_path / folder_name
            target_folder.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    organize_files()