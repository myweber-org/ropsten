
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

class TempFileCleaner:
    def __init__(self, target_dir: Optional[str] = None):
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        self.removed_files = []
        self.removed_dirs = []

    def scan_temp_files(self, patterns: List[str] = None) -> List[Path]:
        if patterns is None:
            patterns = ['*.tmp', 'temp_*', '~*', '*.cache']
        
        found_files = []
        for pattern in patterns:
            found_files.extend(self.target_dir.glob(pattern))
        
        return list(set(found_files))

    def cleanup_files(self, max_age_days: int = 7, dry_run: bool = False) -> dict:
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'total_size': 0,
            'errors': []
        }
        
        try:
            for item in self.target_dir.iterdir():
                try:
                    if item.is_file():
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        if mtime < cutoff_time:
                            if not dry_run:
                                size = item.stat().st_size
                                item.unlink()
                                self.removed_files.append(item)
                                stats['files_removed'] += 1
                                stats['total_size'] += size
                            else:
                                stats['files_removed'] += 1
                    
                    elif item.is_dir() and item.name.startswith('tmp'):
                        if not dry_run:
                            shutil.rmtree(item)
                            self.removed_dirs.append(item)
                            stats['dirs_removed'] += 1
                        else:
                            stats['dirs_removed'] += 1
                            
                except (PermissionError, OSError) as e:
                    stats['errors'].append(f"Failed to remove {item}: {str(e)}")
        
        except Exception as e:
            stats['errors'].append(f"Scan error: {str(e)}")
        
        return stats

    def get_summary(self) -> str:
        total_files = len(self.removed_files)
        total_dirs = len(self.removed_dirs)
        return f"Cleaned {total_files} files and {total_dirs} directories from {self.target_dir}"

def main():
    cleaner = TempFileCleaner()
    print(f"Scanning temporary directory: {cleaner.target_dir}")
    
    result = cleaner.cleanup_files(dry_run=True)
    print(f"Dry run would remove: {result['files_removed']} files, {result['dirs_removed']} directories")
    
    if result['files_removed'] > 0 or result['dirs_removed'] > 0:
        confirm = input("Proceed with cleanup? (y/n): ")
        if confirm.lower() == 'y':
            result = cleaner.cleanup_files(dry_run=False)
            print(cleaner.get_summary())
            if result['errors']:
                print(f"Encountered {len(result['errors'])} errors during cleanup")
    else:
        print("No old temporary files found")

if __name__ == "__main__":
    main()import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory: str, extensions: tuple = ('.tmp', '.temp', '.log'), max_age_days: int = 7):
    """
    Remove temporary files with specified extensions older than a given number of days.
    
    Args:
        directory: Path to the directory to clean.
        extensions: Tuple of file extensions to consider as temporary.
        max_age_days: Maximum age of files in days before they are removed.
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    current_time = os.path.getctime if hasattr(os.path, 'getctime') else os.path.getmtime
    cutoff_time = current_time - (max_age_days * 24 * 60 * 60)

    removed_count = 0
    total_size = 0

    for item in dir_path.rglob('*'):
        if item.is_file() and item.suffix.lower() in extensions:
            try:
                file_time = os.path.getmtime(item)
                if file_time < cutoff_time:
                    file_size = item.stat().st_size
                    item.unlink()
                    removed_count += 1
                    total_size += file_size
                    print(f"Removed: {item.name} ({file_size} bytes)")
            except (OSError, PermissionError) as e:
                print(f"Error removing {item}: {e}")

    print(f"Cleaning complete. Removed {removed_count} files, freed {total_size} bytes.")

def create_test_environment():
    """Create a test directory with temporary files for demonstration."""
    test_dir = tempfile.mkdtemp(prefix='clean_test_')
    print(f"Created test directory: {test_dir}")

    extensions = ['.tmp', '.temp', '.log', '.bak']
    for i in range(10):
        ext = extensions[i % len(extensions)]
        test_file = Path(test_dir) / f"test_file_{i}{ext}"
        test_file.write_text(f"Temporary content {i}")
    
    return test_dir

if __name__ == "__main__":
    try:
        test_env = create_test_environment()
        clean_temp_files(test_env, extensions=('.tmp', '.temp', '.log'), max_age_days=0)
        shutil.rmtree(test_env)
    except Exception as e:
        print(f"An error occurred: {e}")