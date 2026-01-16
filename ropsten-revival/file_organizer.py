
import os
import hashlib
import shutil
from datetime import datetime
import logging

def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating hash for {filepath}: {e}")
        return None

def organize_files(source_dir, target_dir):
    """Organize files by extension and detect duplicates"""
    
    if not os.path.exists(source_dir):
        logging.error(f"Source directory does not exist: {source_dir}")
        return
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    file_hashes = {}
    duplicates = []
    
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            source_path = os.path.join(root, filename)
            
            if not os.path.isfile(source_path):
                continue
            
            file_hash = calculate_file_hash(source_path)
            if file_hash is None:
                continue
            
            if file_hash in file_hashes:
                duplicates.append(source_path)
                logging.info(f"Duplicate found: {source_path}")
                continue
            
            file_hashes[file_hash] = source_path
            
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext:
                file_ext = "no_extension"
            
            ext_dir = os.path.join(target_dir, file_ext[1:])
            if not os.path.exists(ext_dir):
                os.makedirs(ext_dir)
            
            target_path = os.path.join(ext_dir, filename)
            
            counter = 1
            while os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                target_path = os.path.join(ext_dir, f"{name}_{counter}{ext}")
                counter += 1
            
            try:
                shutil.move(source_path, target_path)
                logging.info(f"Moved: {source_path} -> {target_path}")
            except Exception as e:
                logging.error(f"Error moving {source_path}: {e}")
    
    return duplicates

def setup_logging():
    """Configure logging settings"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"file_organizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to organize files"""
    setup_logging()
    
    source_directory = input("Enter source directory path: ").strip()
    target_directory = input("Enter target directory path: ").strip()
    
    if not source_directory or not target_directory:
        print("Both directories must be specified")
        return
    
    print(f"Organizing files from {source_directory} to {target_directory}")
    print("This may take a while depending on the number of files...")
    
    duplicates = organize_files(source_directory, target_directory)
    
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate files:")
        for dup in duplicates:
            print(f"  - {dup}")
    else:
        print("\nNo duplicate files found")
    
    print("\nFile organization completed!")

if __name__ == "__main__":
    main()