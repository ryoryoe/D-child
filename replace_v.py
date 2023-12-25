import csv
import os

# CSVファイルから速度データを読み込む
def load_velocity_data(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
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

    # 各Uファイルを処理
    for filename in os.listdir(U_files_directory):
        if filename.startswith('U'):
            print(1)
            U_file = os.path.join(U_files_directory, filename)
            replace_velocity_in_U_file(U_file, velocity_data)

# CSVファイルとUファイルが含まれるディレクトリのパスを指定
csv_file_path = 'estimate_test_0000.csv'
U_files_directory_path = 'velocity_x=0.10_velocity_y=0.10_predict/20'

main(csv_file_path, U_files_directory_path)
