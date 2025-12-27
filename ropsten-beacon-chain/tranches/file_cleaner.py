
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
    main()