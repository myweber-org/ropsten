
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