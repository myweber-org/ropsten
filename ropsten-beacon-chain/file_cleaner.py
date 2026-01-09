
import argparse
import sys

def remove_duplicates(input_file, output_file):
    seen = set()
    try:
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
        
        with open(output_file, 'w') as outfile:
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    outfile.write(line)
        
        print(f"Successfully removed duplicates. Original lines: {len(lines)}, Unique lines: {len(seen)}")
        return True
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except IOError as e:
        print(f"Error processing files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate lines from a text file.')
    parser.add_argument('input', help='Path to the input file')
    parser.add_argument('output', help='Path to the output file')
    
    args = parser.parse_args()
    
    if not remove_duplicates(args.input, args.output):
        sys.exit(1)

if __name__ == "__main__":
    main()import os
import re
import sys

def normalize_filename(filename):
    """Convert filename to lowercase, replace spaces with underscores, and remove special chars."""
    name, ext = os.path.splitext(filename)
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name + ext.lower()

def clean_directory(directory_path):
    """Rename all files in the directory with normalized names."""
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return False

    for filename in os.listdir(directory_path):
        old_path = os.path.join(directory_path, filename)
        if os.path.isfile(old_path):
            new_name = normalize_filename(filename)
            new_path = os.path.join(directory_path, new_name)
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                except OSError as e:
                    print(f"Failed to rename {filename}: {e}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_cleaner.py <directory_path>")
        sys.exit(1)
    target_dir = sys.argv[1]
    clean_directory(target_dir)
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

class TemporaryFileCleaner:
    def __init__(self, target_dir: Optional[str] = None):
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        self.supported_extensions = {'.tmp', '.temp', '.log', '.cache'}
        self.max_age_days = 7

    def scan_directory(self) -> List[Path]:
        found_files = []
        for item in self.target_dir.rglob('*'):
            if item.is_file():
                if item.suffix.lower() in self.supported_extensions:
                    found_files.append(item)
        return found_files

    def is_old_file(self, file_path: Path) -> bool:
        import time
        current_time = time.time()
        file_mtime = file_path.stat().st_mtime
        age_days = (current_time - file_mtime) / (24 * 3600)
        return age_days > self.max_age_days

    def cleanup(self, dry_run: bool = False) -> dict:
        results = {
            'scanned': 0,
            'removed': 0,
            'failed': 0,
            'details': []
        }
        
        try:
            files = self.scan_directory()
            results['scanned'] = len(files)
            
            for file_path in files:
                try:
                    if self.is_old_file(file_path):
                        if not dry_run:
                            file_path.unlink()
                            results['removed'] += 1
                            results['details'].append(f"Removed: {file_path}")
                        else:
                            results['details'].append(f"Would remove: {file_path}")
                except Exception as e:
                    results['failed'] += 1
                    results['details'].append(f"Error with {file_path}: {str(e)}")
                    
        except Exception as e:
            results['details'].append(f"Scan error: {str(e)}")
            
        return results

    def set_extensions(self, extensions: set):
        self.supported_extensions = {ext.lower() for ext in extensions}

def main():
    cleaner = TemporaryFileCleaner()
    print(f"Scanning directory: {cleaner.target_dir}")
    
    results = cleaner.cleanup(dry_run=True)
    
    print(f"\nScan Results:")
    print(f"Files scanned: {results['scanned']}")
    print(f"Files to remove: {results['removed']}")
    print(f"Errors: {results['failed']}")
    
    if input("\nProceed with cleanup? (y/n): ").lower() == 'y':
        results = cleaner.cleanup(dry_run=False)
        print(f"\nCleanup completed. Removed {results['removed']} files.")

if __name__ == "__main__":
    main()