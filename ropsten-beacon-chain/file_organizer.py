
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
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
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(source_dir, target_base_dir=None):
    """
    Organize files in source_dir into subdirectories based on file extensions.
    If target_base_dir is not provided, uses source_dir as base.
    """
    if target_base_dir is None:
        target_base_dir = source_dir
    
    source_path = Path(source_dir)
    target_base_path = Path(target_base_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist.")
        return
    
    if not target_base_path.exists():
        target_base_path.mkdir(parents=True, exist_ok=True)
    
    extension_categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tiff'],
        'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'],
        'archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.json', '.xml'],
        'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
    }
    
    other_dir = target_base_path / 'other'
    other_dir.mkdir(exist_ok=True)
    
    for item in source_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False
            
            for category, extensions in extension_categories.items():
                if file_ext in extensions:
                    category_dir = target_base_path / category
                    category_dir.mkdir(exist_ok=True)
                    
                    try:
                        shutil.move(str(item), str(category_dir / item.name))
                        print(f"Moved {item.name} to {category}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            if not moved:
                try:
                    shutil.move(str(item), str(other_dir / item.name))
                    print(f"Moved {item.name} to other/")
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
    
    print("File organization completed.")

def main():
    import sys
    
    if len(sys.argv) > 1:
        source_directory = sys.argv[1]
        target_directory = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        source_directory = input("Enter source directory path: ").strip()
        target_input = input("Enter target directory path (press Enter to use source): ").strip()
        target_directory = target_input if target_input else None
    
    organize_files(source_directory, target_directory)

if __name__ == "__main__":
    main()
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext:
                target_dir = os.path.join(directory, file_ext[1:])
            else:
                target_dir = os.path.join(directory, "no_extension")

            os.makedirs(target_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved: {filename} -> {target_dir}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.move(file_path, os.path.join(target_dir, filename))

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ")
    if os.path.exists(target_directory):
        organize_files(target_directory)
        print("Files organized successfully.")
    else:
        print("Directory does not exist.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subdirectories
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
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

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)