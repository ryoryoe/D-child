#指定したディれクリ内の不要な重みファイルとチェックポイントファイルを削除するスクリプト
#末尾が0でない重みファイルと、save_temp_weightディレクトリ内の最大epoch以外のチェックポイントファイルを削除

import os
import re

def delete_files_with_nonzero_last_digit(base_dir):
    # ファイル名の10の位または1の位が0でないファイルを削除 (save_temp_weightディレクトリを除外)
    for root, dirs, files in os.walk(base_dir):
        # save_temp_weightディレクトリを除外
        dirs[:] = [d for d in dirs if d != 'save_temp_weight']
        
        for file in files:
            # ファイル名の末尾が数字である場合に、その10の位と1の位を確認
            match = re.search(r'(\d+)\.pth$', file)
            if match:
                # 数字部分を取得し、10の位と1の位を確認
                number = int(match.group(1))
                last_digit = number % 10  # 1の位
                second_last_digit = (number // 10) % 10  # 10の位
                
                if last_digit != 0 or second_last_digit != 0:
                    # 10の位または1の位が0でないファイルを削除
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

def delete_non_max_checkpoint_files(base_dir):
    # save_temp_weightディレクトリ内の最大epoch以外のcheckpointファイルを削除
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == 'save_temp_weight':
                dir_path = os.path.join(root, dir_name)
                checkpoint_files = []

                # save_temp_weight内のファイルを取得
                for file in os.listdir(dir_path):
                    match = re.search(r'epoch=(\d+)\.pth$', file)
                    if match:
                        checkpoint_files.append((int(match.group(1)), file))

                # 最大のepoch番号を持つファイルを除いて削除
                if checkpoint_files:
                    max_checkpoint = max(checkpoint_files, key=lambda x: x[0])
                    for epoch, file in checkpoint_files:
                        if epoch != max_checkpoint[0]:
                            file_path = os.path.join(dir_path, file)
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")

# メインディレクトリを指定して関数を実行
base_dir = '/mnt/data1/tony/result'
delete_files_with_nonzero_last_digit(base_dir)
delete_non_max_checkpoint_files(base_dir)


