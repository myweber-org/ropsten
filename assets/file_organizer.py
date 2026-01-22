
import os
import shutil

def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'NoExtension'
            target_folder = os.path.join(directory, file_extension.upper() + "_FILES")
            
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

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension[1:].lower() if extension else 'no_extension'

            target_dir = os.path.join(directory, extension)

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            try:
                shutil.move(file_path, os.path.join(target_dir, filename))
                print(f"Moved '{filename}' to '{extension}/'")
            except Exception as e:
                print(f"Error moving '{filename}': {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(source_dir, target_dir=None):
    if target_dir is None:
        target_dir = source_dir
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist.")
        return
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    extensions_folders = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv'],
        'archives': ['.zip', '.tar', '.gz', '.rar'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    for item in source_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            for folder_name, extensions in extensions_folders.items():
                if file_extension in extensions:
                    folder_path = target_path / folder_name
                    folder_path.mkdir(exist_ok=True)
                    
                    try:
                        shutil.move(str(item), str(folder_path / item.name))
                        print(f"Moved {item.name} to {folder_name}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            if not moved:
                other_folder = target_path / 'other'
                other_folder.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(other_folder / item.name))
                    print(f"Moved {item.name} to other/")
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
    
    print("File organization completed.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        source_directory = sys.argv[1]
        target_directory = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        source_directory = input("Enter source directory path: ")
        target_input = input("Enter target directory path (press Enter to use source): ")
        target_directory = target_input if target_input.strip() else None
    
    organize_files(source_directory, target_directory)