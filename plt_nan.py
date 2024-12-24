import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mymodule import file_maker

#役割
#発散したファイルを区間ごとに読み込んでカラーでプロットする


#input_path = "/mnt/train_data_ver11/delete"
file_name = "3d_flow_mesh16_more_low_divergence"
input_path = f"/mnt/data1/tony/{file_name}/delete"
output_path = "/mnt/data1/tony/plt_nan"
output_file_name = f"nan_velocity_{file_name}_deleted"

#グラフの範囲
min_x = -0.8
max_x = 0.8
min_y = 0.4
max_y = 0.8

border_x = [-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]
#border_y = [0.9, 1.0, 1.1, 1.2,1.3]
border_y = [0.5, 0.6, 0.7, 0.8]
#ここまで毎回設定する

#colors = ['red', 'blue', 'green', 'yellow',"purple","orange","pink","lightblue"]
velocities = []
file_maker(output_path)
# 色の数
num_colors = len(border_x)*len(border_y)
# 色相を0から1まで均等に分割
hue_values = np.linspace(0, 1, num_colors)
# HSV色空間を使用してグラデーションを生成し、それをRGBに変換
colors = [plt.cm.hsv(hue) for hue in hue_values]

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
#色分けしてプロット
count = 0
for i in border_x:
    print(f"{i=}")
    for j in border_y:
        target = np.array(borders['border_{}_{}'.format(i,j)])
        if target.shape == (0,):
            continue
        plt.scatter(target[:, 0], target[:, 1], color=colors[count], s=1)
        plt.text((target[:,0].max()+target[:,0].min())/2,(target[:,1].max()+target[:,1].min())/2,f"{len(target)}",fontsize=10,color='black')
        count += 1
plt.xlabel('velocity_x')
plt.ylabel('velocity_y')
plt.xlim(min_x,max_x)
plt.ylim(min_y,max_y)
plt.title(f"total_num={total_num}")
plt.savefig(f'{output_path}/{output_file_name}.png')
