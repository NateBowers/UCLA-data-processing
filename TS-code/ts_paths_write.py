import os
import shutil
import re
import csv

src_root = "."
dst_root = "ts_data"

os.makedirs(dst_root, exist_ok=True)

with open('ts_paths.csv', 'w') as f:
    writer = csv.writer(f)

    for dirpath, dirnames, filenames in os.walk(src_root):
        # Skip the destination directory to avoid copying files from ts_data
        if os.path.abspath(dirpath).startswith(os.path.abspath(dst_root)):
            continue
        for filename in filenames:
            if filename.endswith(".h5") and 'ts' in filename.lower():
                nums = re.findall(r'\d+', filename)
                loc = nums[0]
                volt = nums[1]
                if 'CH' in filename:
                    material = 'CH'
                elif 'Cu' in filename:
                    material = 'Cu'
                elif 'Al' in filename:
                    material = 'Al'
                else:
                    material = ''
                
                filepath = os.path.join(dirpath, filename)

                csv_row = [filepath, material, loc, volt]
                writer.writerow(csv_row)

print('done')

# root = 'ts_data/CH'

# files = os.listdir('ts_data/CH')
# for file in files:
#     numbers = re.findall(r'\d+', file)
#     loc = numbers[0]
#     volt = numbers[1]
#     print(file)
#     print(loc)
#     print(volt)



# print(files)