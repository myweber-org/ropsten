
import sys

def remove_duplicates(input_file, output_file):
    seen = set()
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line not in seen:
                seen.add(line)
                outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        remove_duplicates(input_path, output_path)
        print(f"Duplicates removed. Cleaned file saved as: {output_path}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory: str, extensions: tuple = ('.tmp', '.temp', '.log'), days_old: int = 7):
    """
    Remove temporary files with specified extensions older than a given number of days.
    """
    from datetime import datetime, timedelta
    import time

    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    removed_count = 0
    total_size = 0

    for file_path in target_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            file_stat = file_path.stat()
            if file_stat.st_mtime < cutoff_time:
                try:
                    total_size += file_stat.st_size
                    file_path.unlink()
                    removed_count += 1
                    print(f"Removed: {file_path}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    print(f"Cleaned {removed_count} files, freed {total_size / (1024*1024):.2f} MB")

def create_sample_temp_files():
    """Create sample temporary files for testing."""
    temp_dir = tempfile.mkdtemp()
    print(f"Created test directory: {temp_dir}")

    test_files = [
        "cache.tmp",
        "backup.temp",
        "error.log",
        "data.txt",
        "session.tmp"
    ]

    for fname in test_files:
        file_path = os.path.join(temp_dir, fname)
        with open(file_path, 'w') as f:
            f.write("Sample content for testing cleanup.\n")
        # Set modification time to 10 days ago
        old_time = time.time() - (10 * 24 * 60 * 60)
        os.utime(file_path, (old_time, old_time))

    return temp_dir

if __name__ == "__main__":
    # Example usage
    test_dir = create_sample_temp_files()
    try:
        clean_temp_files(test_dir, extensions=('.tmp', '.temp', '.log'), days_old=5)
    finally:
        # Clean up test directory
        shutil.rmtree(test_dir)
        print(f"Removed test directory: {test_dir}")