
import os
import shutil

def organize_files(directory):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_extension = filename.split('.')[-1]
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            source_path = os.path.join(directory, filename)
            target_path = os.path.join(target_dir, filename)
            shutil.move(source_path, target_path)
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ")
    if os.path.exists(target_directory):
        organize_files(target_directory)
    else:
        print("Directory does not exist.")