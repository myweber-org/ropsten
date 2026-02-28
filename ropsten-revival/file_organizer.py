
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    # Define categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']
    }

    # Ensure the directory exists
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        return

    # Create category folders if they don't exist
    for category in categories:
        category_path = dir_path / category
        category_path.mkdir(exist_ok=True)

    # Track moved files and unknown extensions
    moved_files = []
    unknown_extensions = set()

    # Iterate over files in the directory
    for item in dir_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False

            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    target_path = dir_path / category / item.name
                    # Handle naming conflicts
                    counter = 1
                    while target_path.exists():
                        stem = item.stem
                        new_name = f"{stem}_{counter}{item.suffix}"
                        target_path = dir_path / category / new_name
                        counter += 1

                    # Move the file
                    shutil.move(str(item), str(target_path))
                    moved_files.append((item.name, category))
                    moved = True
                    break

            # If no category matched, record the unknown extension
            if not moved:
                unknown_extensions.add(file_extension)

    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

    if unknown_extensions:
        print(f"\nFound files with unknown extensions: {', '.join(sorted(unknown_extensions))}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)