import os

folder_path = r'C:\Users\phaml\Downloads\data_download\output'
files = os.listdir(folder_path)
files.sort() 

for i, filename in enumerate(files):
    file_ext = os.path.splitext(filename)[1]
    new_name = f"test_{i+1}{file_ext}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
