import os
import shutil

source = 'dataset'
dest = 'small_dataset'

if not os.path.exists(dest):
    os.makedirs(dest)
else:
    print('File already exists')

directories = os.listdir(source)

for dir in directories:
    # print(dir)
    os.makedirs(os.path.join(dest,dir))

for dir in directories:
    list_of_files = os.listdir(os.path.join(source,dir))
    for i in range(50):
        # print(list_of_files[i])
        shutil.copy(os.path.join(source,dir,list_of_files[i]),os.path.join(dest,dir))






