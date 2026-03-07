
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
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
        'video': ['.mp4', '.avi', '.mov', '.mkv', '.flv'],
        'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c'],
        'executables': ['.exe', '.msi', '.sh', '.bat']
    }
    
    # Create category folders if they don't exist
    for category in categories.keys():
        category_path = path / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files and unknown extensions
    moved_files = []
    unknown_extensions = set()
    
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
                        stem = item.stem
                        new_name = f"{stem}_{counter}{item.suffix}"
                        destination = path / category / new_name
                        counter += 1
                    
                    shutil.move(str(item), str(destination))
                    moved_files.append((item.name, category))
                    moved = True
                    break
            
            # If no category found, move to 'other' folder
            if not moved:
                other_folder = path / 'other'
                other_folder.mkdir(exist_ok=True)
                
                destination = other_folder / item.name
                counter = 1
                while destination.exists():
                    stem = item.stem
                    new_name = f"{stem}_{counter}{item.suffix}"
                    destination = other_folder / new_name
                    counter += 1
                
                shutil.move(str(item), str(destination))
                moved_files.append((item.name, 'other'))
                unknown_extensions.add(file_extension)
    
    # Print summary
    print(f"Organization complete for: {directory_path}")
    print(f"Total files processed: {len(moved_files)}")
    
    if moved_files:
        print("\nFiles moved:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if unknown_extensions:
        print(f"\nUnknown extensions encountered: {', '.join(sorted(unknown_extensions))}")
        print("These files were moved to the 'other' folder.")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)