
import os
import shutil
from datetime import datetime
from pathlib import Path
import hashlib

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def organize_files_by_date(source_dir, target_base_dir):
    """
    Organize files from source directory into date-based folders in target directory
    """
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return
    
    os.makedirs(target_base_dir, exist_ok=True)
    
    file_hashes = {}
    organized_count = 0
    duplicate_count = 0
    
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        
        if not os.path.isfile(source_path):
            continue
        
        try:
            # Get file modification time
            mod_time = os.path.getmtime(source_path)
            mod_date = datetime.fromtimestamp(mod_time)
            
            # Create date-based folder structure: YYYY/MM/DD
            date_folder = os.path.join(
                target_base_dir,
                str(mod_date.year),
                f"{mod_date.month:02d}",
                f"{mod_date.day:02d}"
            )
            os.makedirs(date_folder, exist_ok=True)
            
            # Check for duplicates
            file_hash = calculate_file_hash(source_path)
            if file_hash and file_hash in file_hashes:
                print(f"Duplicate found: {item}")
                duplicate_count += 1
                continue
            
            # Move file to date-based folder
            target_path = os.path.join(date_folder, item)
            
            # Handle filename conflicts
            counter = 1
            name_parts = os.path.splitext(item)
            while os.path.exists(target_path):
                new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                target_path = os.path.join(date_folder, new_name)
                counter += 1
            
            shutil.move(source_path, target_path)
            organized_count += 1
            
            if file_hash:
                file_hashes[file_hash] = target_path
                
        except Exception as e:
            print(f"Error processing {item}: {e}")
    
    print(f"Organization complete. Organized {organized_count} files, found {duplicate_count} duplicates.")

def create_summary_report(target_base_dir):
    """
    Create a summary report of organized files
    """
    report_path = os.path.join(target_base_dir, "organization_report.txt")
    
    with open(report_path, 'w') as report:
        report.write("File Organization Report\n")
        report.write("=" * 50 + "\n")
        report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        total_files = 0
        for root, dirs, files in os.walk(target_base_dir):
            # Skip the report file itself
            if "organization_report.txt" in files:
                files.remove("organization_report.txt")
            
            if files:
                rel_path = os.path.relpath(root, target_base_dir)
                report.write(f"Folder: {rel_path}\n")
                report.write(f"  Files: {len(files)}\n")
                for file in sorted(files)[:10]:  # List first 10 files
                    report.write(f"    - {file}\n")
                if len(files) > 10:
                    report.write(f"    ... and {len(files) - 10} more files\n")
                report.write("\n")
                total_files += len(files)
        
        report.write(f"\nTotal files organized: {total_files}\n")
    
    print(f"Summary report created: {report_path}")

if __name__ == "__main__":
    # Example usage
    source_directory = "/path/to/source/files"
    target_directory = "/path/to/organized/files"
    
    # Organize files
    organize_files_by_date(source_directory, target_directory)
    
    # Create summary report
    create_summary_report(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
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
            
            target_path = os.path.join(target_folder, item)
            
            try:
                shutil.move(item_path, target_path)
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)