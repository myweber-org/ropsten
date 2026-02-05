
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
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

class TemporaryFileCleaner:
    """Utility class to identify and remove temporary files."""
    
    TEMP_EXTENSIONS = {'.tmp', '.temp', '.bak', '.swp', '.swo'}
    TEMP_PREFIXES = ('~$', '~', '.~', '#', '.#')
    
    def __init__(self, target_dir: Optional[str] = None):
        """
        Initialize the cleaner with a target directory.
        
        Args:
            target_dir: Directory to clean. If None, uses current directory.
        """
        self.target_dir = Path(target_dir) if target_dir else Path.cwd()
        
    def find_temp_files(self) -> List[Path]:
        """
        Find all temporary files in the target directory.
        
        Returns:
            List of Path objects for temporary files.
        """
        temp_files = []
        
        for file_path in self.target_dir.rglob('*'):
            if file_path.is_file():
                if self._is_temp_file(file_path):
                    temp_files.append(file_path)
        
        return temp_files
    
    def _is_temp_file(self, file_path: Path) -> bool:
        """
        Check if a file is a temporary file.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if the file is a temporary file, False otherwise.
        """
        filename = file_path.name
        
        # Check for temporary extensions
        if file_path.suffix.lower() in self.TEMP_EXTENSIONS:
            return True
        
        # Check for temporary prefixes
        if filename.startswith(self.TEMP_PREFIXES):
            return True
        
        # Check for common temporary file patterns
        if filename.endswith('~') or '.swp' in filename:
            return True
        
        return False
    
    def clean_temp_files(self, dry_run: bool = True) -> dict:
        """
        Remove temporary files from the target directory.
        
        Args:
            dry_run: If True, only list files without deleting.
            
        Returns:
            Dictionary with results of the operation.
        """
        temp_files = self.find_temp_files()
        results = {
            'total_found': len(temp_files),
            'deleted': [],
            'skipped': [],
            'errors': []
        }
        
        for file_path in temp_files:
            try:
                if dry_run:
                    results['skipped'].append(str(file_path))
                else:
                    file_path.unlink()
                    results['deleted'].append(str(file_path))
            except Exception as e:
                results['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        return results
    
    def create_test_temp_files(self, count: int = 5) -> List[Path]:
        """
        Create test temporary files for demonstration purposes.
        
        Args:
            count: Number of test files to create.
            
        Returns:
            List of created file paths.
        """
        created_files = []
        
        for i in range(count):
            # Create different types of temporary files
            patterns = [
                f'temp_file_{i}.tmp',
                f'~backup_{i}.txt',
                f'.#lockfile_{i}',
                f'autosave_{i}.bak',
                f'swapfile_{i}.swp'
            ]
            
            for pattern in patterns[:min(count - i, len(patterns))]:
                temp_file = self.target_dir / pattern
                temp_file.write_text(f'Test temporary content for {pattern}')
                created_files.append(temp_file)
        
        return created_files

def main():
    """Example usage of the TemporaryFileCleaner."""
    # Create a temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix='cleaner_test_'))
    print(f"Testing in directory: {test_dir}")
    
    # Initialize cleaner
    cleaner = TemporaryFileCleaner(str(test_dir))
    
    # Create some test temporary files
    print("\nCreating test temporary files...")
    test_files = cleaner.create_test_temp_files(3)
    print(f"Created {len(test_files)} test files")
    
    # Find temporary files
    print("\nFinding temporary files...")
    found_files = cleaner.find_temp_files()
    print(f"Found {len(found_files)} temporary files:")
    for file in found_files:
        print(f"  - {file.name}")
    
    # Dry run - show what would be deleted
    print("\nPerforming dry run...")
    dry_run_results = cleaner.clean_temp_files(dry_run=True)
    print(f"Would delete {dry_run_results['total_found']} files")
    
    # Actual cleanup
    print("\nPerforming actual cleanup...")
    cleanup_results = cleaner.clean_temp_files(dry_run=False)
    print(f"Deleted {len(cleanup_results['deleted'])} files")
    
    # Clean up test directory
    shutil.rmtree(test_dir)
    print(f"\nCleaned up test directory: {test_dir}")

if __name__ == '__main__':
    main()import os
import time
import logging
from pathlib import Path

def clean_old_files(directory_path, days_old=7):
    """
    Remove files in the specified directory that are older than the given days.
    """
    if not os.path.isdir(directory_path):
        logging.error(f"Directory does not exist: {directory_path}")
        return
    
    cutoff_time = time.time() - (days_old * 86400)
    deleted_count = 0
    error_count = 0
    
    for item in Path(directory_path).iterdir():
        try:
            if item.is_file():
                if item.stat().st_mtime < cutoff_time:
                    item.unlink()
                    deleted_count += 1
                    logging.info(f"Deleted: {item}")
        except Exception as e:
            error_count += 1
            logging.error(f"Failed to delete {item}: {e}")
    
    logging.info(f"Cleanup completed. Deleted: {deleted_count}, Errors: {error_count}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    target_dir = "/tmp/test_cleanup"
    clean_old_files(target_dir)
import sys
import os

def remove_duplicates(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".deduped"
    
    seen_lines = set()
    unique_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.rstrip('\n')
                if stripped_line not in seen_lines:
                    seen_lines.add(stripped_line)
                    unique_lines.append(line)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        
        print(f"Successfully removed duplicates. Output saved to '{output_file}'")
        print(f"Original lines: {len(seen_lines) + (len(unique_lines) - len(seen_lines))}")
        print(f"Unique lines: {len(unique_lines)}")
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
    
    remove_duplicates(input_file, output_file)
import sys

def remove_duplicates(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        unique_lines = []
        seen = set()
        
        for line in lines:
            stripped_line = line.rstrip('\n')
            if stripped_line not in seen:
                seen.add(stripped_line)
                unique_lines.append(line)
        
        with open(output_file, 'w') as f:
            f.writelines(unique_lines)
        
        print(f"Removed {len(lines) - len(unique_lines)} duplicate lines.")
        print(f"Unique lines saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    remove_duplicates(input_file, output_file)