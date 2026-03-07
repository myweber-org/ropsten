import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

class TemporaryFileCleaner:
    def __init__(self, target_dir: Optional[str] = None):
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        self.deleted_files = []
        self.deleted_dirs = []

    def identify_temporary_files(self, patterns: List[str] = None) -> List[Path]:
        if patterns is None:
            patterns = ['*.tmp', 'temp_*', '~*', '*.bak']
        
        found_files = []
        for pattern in patterns:
            found_files.extend(self.target_dir.glob(pattern))
        
        return list(set(found_files))

    def clean_files(self, patterns: List[str] = None, dry_run: bool = False) -> dict:
        files_to_clean = self.identify_temporary_files(patterns)
        results = {
            'files_found': len(files_to_clean),
            'files_deleted': 0,
            'bytes_freed': 0,
            'errors': []
        }

        for file_path in files_to_clean:
            try:
                if dry_run:
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    results['bytes_freed'] += file_size
                    results['files_deleted'] += 1
                    continue

                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    self.deleted_files.append(file_path)
                    results['bytes_freed'] += file_size
                    results['files_deleted'] += 1
                elif file_path.is_dir():
                    dir_size = self._calculate_dir_size(file_path)
                    shutil.rmtree(file_path)
                    self.deleted_dirs.append(file_path)
                    results['bytes_freed'] += dir_size
                    results['files_deleted'] += 1
            except Exception as e:
                results['errors'].append(f"Failed to delete {file_path}: {str(e)}")

        return results

    def _calculate_dir_size(self, directory: Path) -> int:
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def get_summary(self) -> str:
        total_items = len(self.deleted_files) + len(self.deleted_dirs)
        return f"Cleaned {total_items} items: {len(self.deleted_files)} files and {len(self.deleted_dirs)} directories"

def main():
    cleaner = TemporaryFileCleaner()
    print(f"Cleaning temporary files in: {cleaner.target_dir}")
    
    result = cleaner.clean_files(dry_run=True)
    print(f"Dry run found {result['files_found']} files to delete")
    print(f"Would free approximately {result['bytes_freed'] / (1024*1024):.2f} MB")
    
    if result['files_found'] > 0:
        confirm = input("Proceed with deletion? (y/n): ")
        if confirm.lower() == 'y':
            result = cleaner.clean_files(dry_run=False)
            print(cleaner.get_summary())
            print(f"Freed {result['bytes_freed'] / (1024*1024):.2f} MB")
            if result['errors']:
                print(f"Encountered {len(result['errors'])} errors")
        else:
            print("Cleanup cancelled")
    else:
        print("No temporary files found to clean")

if __name__ == "__main__":
    main()
import sys
import os

def clean_file(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return False
    
    if output_path is None:
        output_path = input_path + ".cleaned"
    
    seen = set()
    cleaned_lines = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                stripped = line.rstrip('\n')
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    cleaned_lines.append(stripped)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(cleaned_lines))
        
        print(f"Successfully cleaned file. Output saved to '{output_path}'")
        print(f"Removed {len(seen) - len(cleaned_lines)} duplicate lines.")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    clean_file(input_file, output_file)