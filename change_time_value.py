import os
import shutil
import re
#from mymodule import file_maker
import sys
from tqdm import tqdm


"""--役割--
指定したディレクトリ内のサブディレクトリにあるCSVファイルのうち、時間が最後のファイルをファイルをコピーして、ファイル名をT=99に変更する
csvファイルの最大時間を99に指定するのを忘れた時に使う"""

def find_latest_csv(directory):
    latest_file = None
    latest_time = -1
    csv_pattern = re.compile(r'time=(\d+)\.csv')

    for filename in os.listdir(directory):
        match = csv_pattern.search(filename)
        if match:
            time_value = int(match.group(1))
            if time_value == 99:
                break
            if time_value > latest_time:
                latest_time = time_value
                latest_file = filename

    return latest_file, latest_time

def copy_and_rename_file(src_dir, src_file, dest_dir, new_suffix='time=99.csv'):
    src_path = os.path.join(src_dir, src_file)
    dest_file = re.sub(r'time=\d+\.csv$', new_suffix, src_file)
    dest_path = os.path.join(dest_dir, dest_file)
    shutil.copy(src_path, dest_path)
    #print(f'Copied and renamed {src_path} to {dest_path}')

def process_all_directories(root_dir):
    dest_path = os.path.join(root_dir, "Time=99")
    root_dir = os.path.join(root_dir,"results")
    for subdir in tqdm(os.listdir(root_dir),total=len(os.listdir(root_dir))):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            latest_file, _ = find_latest_csv(subdir_path)
            if latest_file:
                copy_and_rename_file(subdir_path, latest_file, dest_path)

# 実行するディレクトリを指定
results_directory = "/mnt/inlet_value2/inlet_size12_120000"
if not os.path.exists(f"{results_directory}/Time=99"):
    os.mkdir(f"{results_directory}/Time=99")
#file_maker(f"{results_directory}/Time=99")
process_all_directories(results_directory)


