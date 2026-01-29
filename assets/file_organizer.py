
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