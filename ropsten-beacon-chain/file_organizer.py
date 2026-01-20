
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    path = Path(directory_path)
    
    # Define categories and their associated extensions
    categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv'],
        'archives': ['.zip', '.rar', '.tar', '.gz'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
    }
    
    # Create category folders if they don't exist
    for category in categories:
        category_path = path / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    skipped_files = []
    
    # Iterate through files in the directory
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    destination = path / category / item.name
                    
                    # Handle duplicate filenames
                    counter = 1
                    while destination.exists():
                        name_parts = item.stem, item.suffix
                        new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                        destination = path / category / new_name
                        counter += 1
                    
                    shutil.move(str(item), str(destination))
                    moved_files.append((item.name, category))
                    moved = True
                    break
            
            if not moved:
                skipped_files.append(item.name)
    
    # Print summary
    print(f"Organization complete for: {directory_path}")
    print(f"Files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if skipped_files:
        print(f"\nSkipped files (unknown extension): {len(skipped_files)}")
        for filename in skipped_files:
            print(f"  {filename}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)