import os

dir_path = os.path.dirname(os.path.abspath(__file__))
for path, dirs, files in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(path, file)
        if file.startswith('._'):
            os.remove(file_path)
            print(f'removed {file_path}')