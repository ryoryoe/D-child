#負の速度のファイルだけを移動させるファイル

import os
import re
import shutil

# 負のvelocity_xのファイルを移動するディレクトリ
root_path = "/mnt/inlet_value1/inlet_size14"
output_path = f'{root_path}/negative_velocity_files'
os.makedirs(output_path,exist_ok = True)
source_directory = f'{root_path}/results'

# カレントディレクトリ内のファイルを確認
for file in os.listdir(source_directory):
    if file.startswith('velocity_x'):
        # velocity_xの値を抽出
        match = re.search(r'velocity_x=(-?\d+\.\d+)', file)
        if match:
            velocity_x = float(match.group(1))
            # velocity_xが負の場合、ファイルを移動
            if velocity_x < 0:
                print(f"Moving {file} to negative_velocity_files/")
                if os.path.exists(f'output_path/{file}'):
                    shutil.rmtree(f'{output_path}/{file}')
                shutil.move(f'{source_directory}/{file}', output_path)
                #shutil.move(file, output_path)
