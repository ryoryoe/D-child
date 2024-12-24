import os
import shutil
import subprocess
import numpy as np
import csv
import sys
import glob
from tqdm import tqdm
import pandas as pd
import datetime
import pytz
import re
import time
from pathlib import Path
#役割 ; 指定したディレクトリのPファイルをcsvに書き出す

decimal = 3 #小数点以下何桁で区切るか
x_data_num = 600 #x方向のデータ数(300)
y_data_num = 200 #y方向のデータ数(200)
start_inlet = 10 #入り口の左端
end_inlet = 22 #入り口の右端
start_outlet = 14 #出口の左端
end_outlet = 18 #出口の右端

case_directory = "/mnt/inlet_value2/inlet_size12_120000_test/results" #docker
#case_directory = "/mnt/inlet_value2/inlet_size12_120000/results" #docker
#case_directory = "/mnt/data1/tony/inlet_value2/inlet_size12_120000/results"

def read_max_U_file(directory): #directory内の最大の数字のディレクトリを探し、そのディレクトリ内のUファイルを読み込む
    """
    指定されたディレクトリ内のサブディレクトリから最大の数を持つサブディレクトリ内の
    特定のファイルを読み込む関数。

    Args:
    directory (str): 基本ディレクトリのパス。
    filename (str): 読み込みたいファイル名。

    Returns:
    str: ファイルの内容。
    """
    max_number = -1
    # サブディレクトリを検索
    for subdir in os.listdir(directory):
        try:
            # サブディレクトリ名を整数に変換
            number = int(subdir)
            if number > max_number:
                max_number = number
        except ValueError:
            # 数字でないディレクトリを無視
            continue
    if max_number != -1:
        return max_number
    else:
        print("数字のディレクトリが見つかりませんでした。")

def extract_pressure(case_directory):
    time_steps = []
    steady_number = read_max_U_file(case_directory)  # 最大時間ステップを取得
    time_steps.append(steady_number)  # 時間ステップリストに追加
    
    for time_step in time_steps:
        pressures = []  # 圧力データを格納するリスト
        p_file_path = os.path.join(case_directory, str(time_step), "p")  # 圧力ファイルのパスを生成
        
        with open(p_file_path, 'r') as file:
            lines = file.readlines()
        
        start_reading = False
        for line in lines:
            if line.startswith("internalField"):  # internalFieldが見つかったら開始
                start_reading = True
                continue
            if start_reading:
                if line.startswith("(") or line.startswith("nonuniform") or line.strip().isdigit():
                    continue  # "(", "nonuniform" またはリストのサイズの行はスキップ
                if line.startswith(")"):
                    break  # データの終わりを示す")"が来たら終了
                pressure_value = float(line.strip())  # 圧力値を取得してリストに追加
                pressures.append([pressure_value])
        
        base_name = os.path.basename(case_directory)  # ディレクトリ名を取得
        csv_output_path = os.path.join(case_directory, f"{base_name}_p_time={time_step}.csv")
        with open(csv_output_path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(['Pressure'])  # ヘッダーを書き込み
            writer.writerows(pressures)  # 圧力データを書き込み
    
    time_steps.pop()  # 最後に追加した要素を削除
    return np.array(pressures)

#対象となるディレクトリ内にある全てのディレクトリのパスを取得
file_paths = os.listdir(f"{case_directory}")
for file_path in tqdm(file_paths, total=len(file_paths)):
    extract_pressure(f"{case_directory}/{file_path}")

