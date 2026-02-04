
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into
    subfolders based on their file extensions.
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
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the specified directory into subfolders based on their extensions.
    Creates folders for images, documents, archives, audio, video, and others.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Define categories and their associated file extensions
    categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg'],
        'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'],
        'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
        'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
    }

    # Create category folders if they don't exist
    for category in categories.keys():
        category_path = os.path.join(directory_path, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

    # Create 'others' folder for uncategorized files
    others_path = os.path.join(directory_path, 'others')
    if not os.path.exists(others_path):
        os.makedirs(others_path)

    # Iterate over all files in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        # Skip directories
        if os.path.isdir(item_path):
            continue

        # Get file extension
        file_extension = Path(item).suffix.lower()

        # Determine the category for the file
        destination_category = 'others'
        for category, extensions in categories.items():
            if file_extension in extensions:
                destination_category = category
                break

        # Define destination path
        destination_path = os.path.join(directory_path, destination_category, item)

        # Move the file to the appropriate folder
        try:
            shutil.move(item_path, destination_path)
            print(f"Moved '{item}' to '{destination_category}' folder.")
        except Exception as e:
            print(f"Error moving '{item}': {e}")

    print("File organization completed.")

if __name__ == "__main__":
    # Specify the directory to organize (current directory in this example)
    target_directory = os.getcwd()
    organize_files_by_extension(target_directory)