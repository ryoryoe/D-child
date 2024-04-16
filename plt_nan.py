import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mymodule import file_maker
#input_path = "/mnt/train_data_ver11/delete"
input_path = "/mnt/data1/tony/train_data_ver7/delete"
output_path = "/mnt/data1/tony/plt_nan"
output_file_name = "nan_velocity_train_data_ver7_deleted"

border_x = [-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]
border_y = [0.2, 0.4, 0.6, 0.8]
#colors = ['red', 'blue', 'green', 'yellow',"purple","orange","pink","lightblue"]
velocities = []
file_maker(output_path)
# 色の数
num_colors = len(border_x)*len(border_y)
# 色相を0から1まで均等に分割
hue_values = np.linspace(0, 1, num_colors)
# HSV色空間を使用してグラデーションを生成し、それをRGBに変換
colors = [plt.cm.hsv(hue) for hue in hue_values]
print(f"{colors=}")
#input_pathのなかにあるcsvファイルのパスを取得
csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
print(f"{csv_files=}")
borders = {}
for i in border_x:
    for j in border_y:
        #空の配列をborder_x*border_yの数だけ作成
        borders['border_{}_{}'.format(i,j)] = []

total_num = 0
break_sine = False
#for文でcsvファイルを一つずつ読み込む
for csv_file in csv_files:
    with open(f"{input_path}/{csv_file}", mode='r') as file:
    # CSVファイルの各行を読み込む
        for i,line in enumerate(file):
            if i == 0:
                continue
            else:
            # 各行からvelocity_xとvelocity_yの値を抽出します。
                parts = line.strip().split('_')
                velocity_x = (parts[1].split('='))
                velocity_x = float(velocity_x[1])
                velocity_y = float(parts[3].split('=')[1])
                total_num += 1
                 # 抽出した値を2次元配列に追加します。
                #velocities.append([velocity_x, velocity_y])
                #border_yの値によってvelocity_x,velocity_yを追加
                for j in range(len(border_x)):
                    if break_sine:
                        break_sine = False
                        break
                    for k in range(len(border_y)):
                        if velocity_x < border_x[j] and velocity_y < border_y[k]:
                            borders['border_{}_{}'.format(border_x[j],border_y[k])].append([velocity_x,velocity_y])
                            break_sine = True
                            break
#velocitiesをnumpy配列に変換
#velocities = np.array(velocities)

#velocitiesをプロット
#点の大きさを小さくする
#plt.scatter(velocities[:, 0], velocities[:, 1], s=1)
#色分けしてプロット
count = 0
for i in border_x:
    print(f"{i=}")
    for j in border_y:
        target = np.array(borders['border_{}_{}'.format(i,j)])
        print(f"{i=},{j=},{len(target)=}")
        print(f"{target[:5]=}")
        print(f"{(target[:,0].max()+target[:,0].min())/2=}")
        plt.scatter(target[:, 0], target[:, 1], color=colors[count], s=1)
        plt.text((target[:,0].max()+target[:,0].min())/2,(target[:,1].max()+target[:,1].min())/2,f"{len(target)}",fontsize=10,color='black')
        plt.xlabel('velocity_x')
        plt.ylabel('velocity_y')
        plt.title(f"total_num={total_num}")
        plt.savefig(f'{output_path}/{output_file_name}.png')
        count += 1
"""for count,i in enumerate(border_y):
    #target = np.array(borders['border_{}'.format(i)])
    target = np.array(borders['border_{}_{}'.format(border_x[0],i)])
    plt.scatter(target[:, 0], target[:, 1], color=colors[count], s=1,label=f"under_{i}={len(target)}")"""
#plt.legend(loc='upper left')
plt.xlabel('velocity_x')
plt.ylabel('velocity_y')
plt.title(f"total_num={total_num}")
plt.savefig(f'{output_path}/{output_file_name}.png')
