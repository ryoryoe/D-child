import numpy as np
import pandas as pd
from mymodule import file_name_maker
import sys
import os

error_list = []
target = "train_data_ver10"
eval_time = 5 #位が変わると取り除く文字数が変わる:wq
input_path = f"../{target}/Time=1"
eval_path = f"../{target}/Time={eval_time}"

input_file_name = file_name_maker(input_path)
eval_file_name = file_name_maker(eval_path)
print(f"{len(input_file_name)=}")
print(f"{len(eval_file_name)=}")

for i in range(len(input_file_name)):
    a = input_file_name[i]
    a = a[:-7]
    input_file_name[i] = a

for i in range(len(eval_file_name)):
    b = eval_file_name[i]
    if eval_time >= 10:
        b = b[:-8]
    else:
        b = b[:-7]
    eval_file_name[i] = b
print("trainファイルにしかないファイル名")
for i in range(len(input_file_name)):
    if not input_file_name[i] in eval_file_name:
        print(f"{input_file_name[i]=}")
        #当該ファイルを削除する
        #os.remove(f"{input_path}/{input_file_name[i]}*")
        error_list.append(input_file_name[i])
print(f"{len(error_list)=}")

print("evalファイルにしかないファイル名")
error_list = []
for i in range(len(eval_file_name)):
    if not eval_file_name[i] in input_file_name:
        print(f"{eval_file_name[i]=}")
        #当該ファイルを削除する
        #os.remove(f"{eval_path}/{eval_file_name[i]}*") 
        error_list.append(input_file_name[i])
print(f"{len(error_list)=}")

"""if a != b:
    print(f"{input_file_name[i]=}")
    print(f"{eval_file_name[i]=}")
    print(f"{input_file_name[i-1]=}")
    print(f"{eval_file_name[i-1]=}")
    print(f"{input_file_name[i+1]=}")
    print(f"{eval_file_name[i+1]=}")
    print(f"{i=}")
    sys.exit()"""
"""else:
    print(f"{a=}")
    print(f"{b=}")
    if i == 5:
        sys.exit()"""
