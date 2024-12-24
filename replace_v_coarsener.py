import os
import shutil
import sys
import csv
from mymodule import file_maker
from tqdm import tqdm
import threading
import concurrent.futures
import pandas as pd

#粗くした速度の計算結果を置き換える

# Uファイルの速度部分を置き換える
def replace_velocity_in_U_file(U_file, velocity_data):
    with open(U_file, 'r',encoding='ascii') as file:
        lines = file.readlines()

    # internalFieldセクションを見つける
    start_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('internalField'):
            start_index = i + 3  # internalField定義の3行後にデータ開始
            break
    if start_index == -1:
        raise Exception('internalField section not found in U file')

    # 速度データの置き換え
    for i, velocity in enumerate(velocity_data):
        lines[start_index + i] = f"({velocity[0]} {velocity[1]} 0)\n"

    # ファイルに書き戻す
    with open(U_file, 'w',encoding='ascii') as file:
        file.writelines(lines)

input_path = r"/Users/tanakaryo/Documents/openfoam/obstacle_flow_test/meshcoarser_test"
csv_path = r"/Users/tanakaryo/Documents/openfoam/obstacle_flow_test/cube_obstacle_two_dimension2"

velocity_data = pd.read_csv(os.path.join(csv_path,"coarse_mesh.csv"),usecols=[2,3]).values

replace_velocity_in_U_file(os.path.join(input_path, '20', 'U'), velocity_data)
