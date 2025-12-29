
import os
import hashlib
import shutil
import logging
from datetime import datetime
from pathlib import Path

class FileOrganizer:
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.duplicate_dir = self.target_dir / "duplicates"
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('file_organizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_hash(self, filepath):
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def organize_by_extension(self):
        """Organize files by their extensions"""
        self.logger.info(f"Starting organization of {self.source_dir}")
        
        if not self.source_dir.exists():
            self.logger.error(f"Source directory {self.source_dir} does not exist")
            return
        
        self.target_dir.mkdir(exist_ok=True)
        self.duplicate_dir.mkdir(exist_ok=True)
        
        file_hashes = {}
        processed_count = 0
        duplicate_count = 0
        
        for item in self.source_dir.rglob('*'):
            if item.is_file():
                try:
                    file_hash = self.calculate_hash(item)
                    
                    if file_hash in file_hashes:
                        duplicate_count += 1
                        duplicate_path = self.duplicate_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{item.name}"
                        shutil.move(str(item), str(duplicate_path))
                        self.logger.warning(f"Duplicate found: {item.name} moved to duplicates folder")
                    else:
                        file_hashes[file_hash] = True
                        extension = item.suffix[1:] if item.suffix else "no_extension"
                        category_dir = self.target_dir / extension
                        category_dir.mkdir(exist_ok=True)
                        
                        target_path = category_dir / item.name
                        if target_path.exists():
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            target_path = category_dir / f"{item.stem}_{timestamp}{item.suffix}"
                        
                        shutil.move(str(item), str(target_path))
                        processed_count += 1
                        self.logger.info(f"Moved {item.name} to {extension} folder")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {item.name}: {str(e)}")
        
        self.logger.info(f"Organization complete. Processed: {processed_count}, Duplicates: {duplicate_count}")
        return {
            'processed': processed_count,
            'duplicates': duplicate_count,
            'timestamp': datetime.now().isoformat()
        }

def main():
    organizer = FileOrganizer(
        source_dir="~/Downloads",
        target_dir="~/Documents/OrganizedFiles"
    )
    
    results = organizer.organize_by_extension()
    
    print(f"Organization Results:")
    print(f"Files Processed: {results['processed']}")
    print(f"Duplicates Found: {results['duplicates']}")
    print(f"Completed at: {results['timestamp']}")

if __name__ == "__main__":
    main()
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