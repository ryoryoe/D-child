#役割 推定済みのファイルをみてcsvファイルからフォルダ名を取得してそのフォルダだけをコピーする
import os
import shutil

# 入力ディレクトリ、検索するディレクトリ、出力ディレクトリを指定
input_dir = "/mnt/result/inlet_value1_1212_transformer/train_data_ver11"  # ここを適宜変更
search_dir = "/mnt/train_data_ver11/results"  # ここを適宜変更
output_dir = "/mnt/train_data_ver11_for_inlet_value1_1212_transformer/results"  # ここを適宜変更
#input_dir = "/mnt/data1/tony/result/inlet_value1_1212_transformer/train_data_ver11"  # ここを適宜変更
#search_dir = "/mnt/data1/tony/train_data_ver11/results"  # ここを適宜変更
#output_dir = "/mnt/data1/tony/train_data_ver11_for_inlet_value1_1212_transformer/results"  # ここを適宜変更

# コピー先ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# 条件に一致するファイル名を収集
def extract_matching_files(input_dir):
    matching_files = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            # ファイル名の一部を抽出
            parts = file_name.split("_")
            velocity_part = "_".join(parts[1:3])  # estimate_velocity_xとvelocity_y部分
            matching_files.append(velocity_part)
    return matching_files

# 指定ディレクトリ内の条件に一致するディレクトリを探してコピー
def copy_matching_dirs(search_dir, matching_files, output_dir):
    for dir_name in os.listdir(search_dir):
        dir_path = os.path.join(search_dir, dir_name)
        if os.path.isdir(dir_path):  # ディレクトリであることを確認
            for match in matching_files:
                if match in dir_name:  # ディレクトリ名に一致するか確認
                    shutil.copytree(dir_path, os.path.join(output_dir, dir_name), dirs_exist_ok=True)
                    print(f"Copied: {dir_name}")
                    break

# 条件に一致するファイル名を取得
matching_files = extract_matching_files(input_dir)

# 一致するディレクトリをコピー
copy_matching_dirs(search_dir, matching_files, output_dir)

print("Completed.")

