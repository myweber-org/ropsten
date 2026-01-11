
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            folder_name = file_extension.upper() + '_FILES'
            folder_path = os.path.join(directory, folder_name)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            shutil.move(file_path, os.path.join(folder_path, filename))
            print(f"Moved {filename} to {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)