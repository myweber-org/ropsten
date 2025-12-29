
import os
import sys
from datetime import datetime

def rename_files_by_date(directory, prefix="file_"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            stat = os.stat(filepath)
            creation_time = datetime.fromtimestamp(stat.st_ctime)
            new_name = f"{prefix}{creation_time.strftime('%Y%m%d_%H%M%S')}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(directory, new_name)
            
            counter = 1
            while os.path.exists(new_path):
                name_part, ext = os.path.splitext(new_name)
                new_path = os.path.join(directory, f"{name_part}_{counter}{ext}")
                counter += 1
            
            os.rename(filepath, new_path)
            print(f"Renamed: {filename} -> {os.path.basename(new_path)}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        prefix = sys.argv[2] if len(sys.argv) > 2 else "file_"
    else:
        target_dir = input("Enter directory path: ")
        prefix = input("Enter filename prefix (default 'file_'): ") or "file_"
    
    if os.path.isdir(target_dir):
        rename_files_by_date(target_dir, prefix)
    else:
        print(f"Directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)