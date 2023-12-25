import os
import shutil
import sys
import csv
from mymodule import file_maker
from tqdm import tqdm

# CSVファイルから速度データを読み込む
def load_velocity_data(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        velocity_data = [(float(row[0]), float(row[1])) for row in reader]
    return velocity_data

# Uファイルの速度部分を置き換える
def replace_velocity_in_U_file(U_file, velocity_data):
    with open(U_file, 'r') as file:
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
    with open(U_file, 'w') as file:
        file.writelines(lines)

# メイン処理
def main(csv_file, U_files_directory):
    velocity_data = load_velocity_data(csv_file)
    if not os.path.exists(U_files_directory):
        return 0
    # 各Uファイルを処理
    for filename in os.listdir(U_files_directory):
        if filename.startswith('U'):
            U_file = os.path.join(U_files_directory, filename)
            replace_velocity_in_U_file(U_file, velocity_data)

specified_folder = '../CNN_2D/estimate_result'  # 予測結果のファイル
CFD_path = "../train_data_ver1/results"#CFDを行ったファイル

# 1. 指定したフォルダ内のcsvファイルのパスとファイル名を取得する
csv_files = [f for f in os.listdir(specified_folder) if f.endswith('.csv')]
csv_estimate_path = [os.path.join(specified_folder, file) for file in csv_files]
csv_estimate_file_name = csv_files
# 2. csv_estimate_pathとcsv_estimate_file_nameから、要素を一つ取り出す
for path, name in tqdm(zip(csv_estimate_path, csv_estimate_file_name),total=len(csv_estimate_path)):
    # 3. nameからフォルダ名を抽出し、新たに指定されたフォルダの中の対応するフォルダをコピーする
    folder_name = name.replace("estimate_", "").replace("_time=1.csv", "")  # "velocity_x=0.77_velocity_y=0.10_time=1"の部分を取得
    source_folder_path = os.path.join(f'{CFD_path}', folder_name)  # 新たに指定されたフォルダからのパス
    destination_folder_path = os.path.join(specified_folder, folder_name)  # 現在のディレクトリにコピーするパス
    
    # 新たなフォルダをコピーする（存在する場合はスキップ）
    if not os.path.exists(destination_folder_path):
        shutil.copytree(source_folder_path, destination_folder_path)
    
    main(path,f"{destination_folder_path}/20")
