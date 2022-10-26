import os
import time
import shutil
dir_path = os.path.dirname("/opt/ml/input/data/train/images/")
loop = 0

"""
dir_path
|_ subdir
     |_ file

을

dir_path
|_(sub_dir+file) 형태로 복사해줌

사용시에는 tmpdir을 제외하고 하면 될듯    
"""

for subdir in os.listdir(dir_path):
    filepath = os.path.join(dir_path, subdir)
    if subdir.endswith("jpg") or subdir.endswith("jpeg"): continue
    for file in os.listdir(filepath):
        old_path = os.path.join(filepath, file)
        new_name = subdir + "_"+ file
        new_path = os.path.join(dir_path, new_name)
        shutil.copy(old_path, new_path)

