#from msilib import add_data
import numpy as np
import csv
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import re
import math
import os
import pickle
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchvision
#import torchvision.transforms as transforms
import copy
import scipy
import pandas as pd
from scipy.optimize import curve_fit
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sys
import torch.nn.functional as F

#parameter
#--------------------------------------------------
#Net1
hidden_dim = 64
lstm_input = 256

batch = 128
#--------------------------------------------------
#function
#--------------------------------------------------
def condition_text(message, output):
    # ファイルを開き、messageの内容を書き込む
    with open(f"../result/{output}/condition.txt", 'w') as file:
        file.write(message)


def sort_and_combine_strings(input_array):
    a = []
    b = []
    for string in input_array:
        if not '-' in string:
            a.append(string)
        if '-' in string:
            b.append(string)
    file_names = a + b
    return file_names

def data_to_csv(path,header_rule=None,end_cut=0,usecol=0):
    if header_rule:
        data =  pd.read_csv(path,dtype=float,usecols=[usecol])
    else:
        data =  pd.read_csv(path,dtype=float,usecols=[usecol],header=None)
    data = data.values
    data = data.astype(float)
    data = np.reshape(data,-1)
    #data = np.delete(data,-1)
    for i in range(end_cut):
        data = np.delete(data,-(i+1))
    return data

def file_maker(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return 0

def file_name_maker(input_path):
    file_names = sorted(os.listdir(input_path))
    file_names = [fn for fn in file_names if "csv" in fn]
    for i in range(len(file_names)):
        file_names[i] = file_names[i].replace(".csv","")
    return file_names


# ファイル処理部分を分離した新しい関数
def process_file(path, eval_path,dmax, dmin):
    df = pd.read_csv(path)
    df = df.values
    df = df.astype(float)   
    df_eval = pd.read_csv(eval_path)
    df_eval = df_eval.values
    df_eval = df_eval.astype(float)
    avg = np.mean(df)
    std = np.std(df)
    avg_eval = np.mean(df_eval)
    std_eval = np.std(df_eval)
    #df -= avg
    #df /= std
    #df_eval -=avg_eval
    #df_eval /= std_eval
    return df,df_eval,avg,std

def round_to_significant_digits(arr, digits=3):
    """
    配列の各要素を指定された有効数字で丸めます。

    :param arr: NumPy配列。
    :param digits: 丸める有効数字の桁数。
    :return: 有効数字で丸められた配列。
    """
    # ゼロチェック
    arr_nonzero = np.where(arr == 0, 1, arr)

    # 指数を計算
    magnitude = np.power(10, digits - np.ceil(np.log10(np.abs(arr_nonzero))))

    # 丸め処理
    return np.round(arr * magnitude) / magnitude

# サンプル配列


# 有効数字3桁で丸める
def Preprocessing(inputname,input_evalname, dmax, dmin,width):
    input_path = sorted(glob.glob(inputname + "/*.csv"), key=natural_keys)
    eval_path = sorted(glob.glob(input_evalname + "/*.csv"), key=natural_keys)
    num_cores = os.cpu_count()
    file_names = file_name_maker(f"{inputname}")
    file_names = sort_and_combine_strings(file_names)
    # ProcessPoolExecutorを使用して各ファイルを並列処理
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_file, input_path,eval_path, [dmax] * len(input_path), [dmin] * len(input_path)),total=len(input_path)))
    # 結果を統合
    train,evals,avg,std = zip(*results)
    train = np.asarray(train, dtype=float)
    evals = np.asarray(evals, dtype=float)
    #train = round_to_significant_digits(train, digits=4)# 並列処理を実装するためのPreprocessing関数
    #evals = round_to_significant_digits(evals, digits=4)# 並列処理を実装するためのPreprocessing関数
    #print(f"{train[:5]=}")
    avg = np.asarray(avg,dtype=float)
    std = np.asarray(std,dtype=float)
    """train = []
    evals = []
    for index in range(len(input_path)):
        df,df_eval = process_file(input_path[index],eval_path[index],0,0)
        train.append(df)
        evals.append(df_eval)
    train = np.asarray(train)
    evals = np.asarray(evals)"""
    # 最終的なデータセットの形状を調整
    train = np.reshape(train, [len(train),width*width,2])
    evals = np.reshape(evals, [len(evals),width*width,2])
    evals = evals[:len(train)]
    file_names = file_names[:len(train)]
    #print(f"before_train={len(train)}")
    #print(f"before_eval={len(evals)}")
    #nan_indices = np.any(np.isnan(array), axis=(1, 2))
    nan_indices = np.where(np.any(np.isnan(evals),axis=(1,2)))[0]
    #train = np.delete(train,nan_indices,axis=0)
    #evals = np.delete(evals,nan_indices,axis=0)
    greater_than_two = np.any(evals >= 2, axis=(1, 2))
    indices_to_remove = np.where(greater_than_two)[0]
    indices_to_remove = np.concatenate([indices_to_remove,nan_indices])
    indices_to_remove = np.sort(indices_to_remove)
    train = np.delete(train,indices_to_remove,axis=0)
    evals = np.delete(evals,indices_to_remove,axis=0)
    avg = np.delete(avg,indices_to_remove,axis=0)
    std = np.delete(std,indices_to_remove,axis=0)
    print(f"delete_index={indices_to_remove}")
    print(f"len(delete_index)={len(indices_to_remove)}")
    #print(f"after_train={len(train)}")
    #print(f"after_eval={len(evals)}") 
    train = np.reshape(train, [-1,width,width,2]).transpose(0,3,1, 2)
    evals = np.reshape(evals, [-1,width,width,2]).transpose(0,3,1,2)
    # リストをNumPy配列に変換
    np_array = np.array(file_names)

    # 削除するインデックス以外を選択してNumPy配列として格納
    filtered_np_array = np.delete(np_array, indices_to_remove)

    #   NumPy配列をリストに変換して出力
    file_names_delete = filtered_np_array.tolist()

    """file_names_delete=[]
    for i in range(len(file_names)):
        if not i in indices_to_remove:
            file_names_delete.append(file_names[i])
            if i<=10:
                print(f"in_file_names[{i}]:{file_names[i]}")
        else:
            if i<=10:
                print(f"out_file_names[{i}]:{file_names[i]=}")
        if i==5:
            print(f"{file_names_delete=}")
            sys.exit()"""
    #file_names_delete = np.delete(file_names,indices_to_remove,axis=0)
    #file_names_delete = [item for idx, item in enumerate(file_names) if idx not in indices_to_remove]
    #file_names_delete = file_names
    return train, evals,file_names_delete,avg,std

        

def atoi(text):
        return int(text) if text.isdigit() else text
def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#---------------------------------------------------

#class
class CNN_2D_mask_0114(nn.Module):
    def __init__(self):
        super(CNN_2D_mask_0114, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(2, 16, kernel_size=9, padding="same")  # Input channels = 2 (x, y velocities), output channels = 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=9, padding="same")  # Increase channels
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling

        # Decoder
        self.conv3 = nn.Conv2d(32, 16, kernel_size=9, padding="same")
        self.conv4 = nn.Conv2d(16, 2, kernel_size=9, padding="same")  # Output channels = 2 to match input dimensions
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsampling
        self.dropout = nn.Dropout(0.25)  

    def forward(self, x):
        # Encoding path
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Downsample
        x = F.relu(self.conv2(x))

        # Decoding path
        x = self.upsample(x)  # Upsample
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.conv4(x)  # No activation, this is the output layer
        return x
class CNN_2D_mask_0113(nn.Module):
    def __init__(self):
        super(CNN_2D_mask_0113, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(2, 16, kernel_size=6, padding="same")  # Input channels = 2 (x, y velocities), output channels = 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=6, padding="same")  # Increase channels
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling

        # Decoder
        self.conv3 = nn.Conv2d(32, 16, kernel_size=6, padding="same")
        self.conv4 = nn.Conv2d(16, 2, kernel_size=6, padding="same")  # Output channels = 2 to match input dimensions
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsampling
        self.dropout = nn.Dropout(0.25)  

    def forward(self, x):
        # Encoding path
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Downsample
        x = F.relu(self.conv2(x))

        # Decoding path
        x = self.upsample(x)  # Upsample
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.conv4(x)  # No activation, this is the output layer
        return x

class New_2DCNN(nn.Module):
    def __init__(self):
        super(New_2DCNN,self).__init__()
        # Encode_r
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)  # Input channels = 2 (x, y velocities), output channels = 16
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Input channels = 2 (x, y velocities), output channels = 16
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Increase channels
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling

        # Decoder
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(8, 2, kernel_size=3, padding=1)  # Output channels = 2 to match input dimensions
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # Upsampling
        self.dropout = nn.Dropout(0.25)  

    def forward(self, x):
        # Encoding path
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Downsample
        x = F.relu(self.conv2(x))

        # Decoding path
        #x = self.upsample(x)  # Upsample
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # Downsample
        x = F.relu(self.conv5(x))

        x = self.upsample(x)  # Upsample
        x = self.dropout(x)
        x = self.conv6(x)  # No activation, this is the output layer
        return x

class Complicated2DCNN(nn.Module):
    def __init__(self):
        super(Complicated2DCNN, self).__init__()

        # Decoder
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # Upsampling
        self.dropout = nn.Dropout(0.25)  
        
        self.conv1 = nn.Conv2d(2, 8,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16,kernel_size=6, stride=1,padding=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,64,kernel_size=6, stride=1,padding=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv4 = nn.Conv2d(64,128,kernel_size=6, stride=1,padding=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256,kernel_size=6,stride=1,padding=3)
        self.bn5= nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=6, padding=3)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=6, padding=3)
        self.conv8 = nn.Conv2d(64, 16, kernel_size=6, padding=3)
        self.conv9 = nn.Conv2d(16, 2, kernel_size=3, padding=1)  # Output channels = 2 to match input dimensions
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling

    def forward(self, x):
        # Encoding path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.upsample(x)  # Upsample
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.upsample(x)  # Upsample
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv6(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.upsample(x)  # Upsample
        x = self.conv7(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv8(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.upsample(x)  # Upsample
        x = self.dropout(x)
        x = self.conv9(x)
        
        #x = self.gap(x)
        return x

class New2DCNN_0105(nn.Module):
    def __init__(self):
        super(New2DCNN_0105, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # Input channels = 2 (x, y velocities), output channels = 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Increase channels
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Input channels = 2 (x, y velocities), output channels = 16
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Input channels = 2 (x, y velocities), output channels = 16
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # Increase channels
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # Increase channels
        # Decoder
        self.conv7 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(16, 2, kernel_size=3, padding=1)  # Output channels = 2 to match input dimensions
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsampling
        self.dropout = nn.Dropout(0.25)  

    def forward(self, x):
        # Encoding path
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Downsample
        x = F.relu(self.conv2(x))

        # Decoding path
        x = self.upsample(x)  # Upsample
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # Downsample
        x = F.relu(self.conv4(x))
        x = self.upsample(x)  # Upsample
        x = F.relu(self.conv5(x))
        x = self.pool(x)  # Downsample

        # Decoding path
        x = F.relu(self.conv6(x))
        x = self.upsample(x)  # Upsample
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)  # No activation, this is the output layer
        return x

class Simple2DCNN_1230(nn.Module):
    def __init__(self):
        super(Simple2DCNN_1230, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # Input channels = 2 (x, y velocities), output channels = 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Increase channels
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling

        # Decoder
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 2, kernel_size=3, padding=1)  # Output channels = 2 to match input dimensions
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsampling
        self.dropout = nn.Dropout(0.25)  

    def forward(self, x):
        # Encoding path
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Downsample
        x = F.relu(self.conv2(x))

        # Decoding path
        x = self.upsample(x)  # Upsample
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.conv4(x)  # No activation, this is the output layer
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 層を定義する
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1,padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.fc = nn.Linear(80000, 20000) 
        # 逆畳み込み（Transpose Convolution）またはアップサンプリング層
        self.up_sample = nn.Upsample(size=10000, mode='linear')
        # 最終出力層
        self.conv_final = nn.Conv1d(32, 2, kernel_size=1)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.up_sample(x)
        x = self.conv_final(x)
        return x

class Net1D(nn.Module):
    def __init__(self):
        super(Net1D,self).__init__()

        self.conv1 = nn.Conv2d(1, 16,kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=6, stride=2)
        self.conv2 = nn.Conv1d(8, 16,kernel_size=6, stride=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16,64,kernel_size=6, stride=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv4 = nn.Conv1d(64,128,kernel_size=6, stride=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128,256,kernel_size=6,stride=1)
        #self.fc = nn.Linear(256,1)
        #self.bn5 = nn.BatchNorm1d(256)
        #self.conv6 = nn.Conv1d(256,512,kernel_size=3,stride=1)
        self.dropout = nn.Dropout(0.25)  
        self.lstm = nn.LSTM(32,64,batch_first=True)
        self.fc = nn.Linear(64,1)
    
    def forward(self,x):
    
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        #x = self.bn4(x)
        #x = self.relu(x)
        #x = self.maxpool(x)
        #x = self.conv5(x)
        #x = self.bn5(x)
        #x = self.relu(x)
        #x = self.maxpool(x)
        #x = self.conv6(x)
        #x = self.gap(x)
        #x = x.view(x.size(0),-1)
        #x = self.dropout(x)
        #x = self.fc(x)
        x = x.view(batch,-1,32)
        #x = x.view(batch,-1,32)
        #print(x.size)
        a,lstm_out = self.lstm(x)
        x = lstm_out[0].view(-1,64)
        x = self.fc(x)
        return x

#def compute_losses(predicted_velocity, true_velocity, inlet_wind_speed):
#    """
#    推定結果からデータ損失、物理損失、壁の損失、入り口の損失を計算し、総合損失を返す関数。
#
#    Parameters:
#    - predicted_velocity: モデルが予測した速度場（バッチサイズ, 2, 32, 32）
#    - true_velocity: 正解の速度場（バッチサイズ, 2, 32, 32）
#    - inlet_wind_speed: 入り口風速の定数値（スカラー）
#
#    Returns:
#    - total_loss: 総合損失
#    - losses: 個別の損失を含む辞書
#    """
#    # 重みの設定
#    data_loss_weight = 5
#    physics_loss_weight = 0.1
#    wall_loss_weight = 0.5
#    inlet_loss_weight = 1.0
#    
#    #初期設定
#    #data_loss_weight = 2.0
#    #physics_loss_weight = 0.1
#    #wall_loss_weight = 0.5
#    #inlet_loss_weight = 0.5
#
#    # メッシュの解像度
#    nx, ny = 32, 32
#    dx = dy = 3.2 / 32  # セルサイズ（0.1m）
#
#    # 1. データ損失（MSE）
#    data_loss = torch.mean((predicted_velocity - true_velocity) ** 2)
#
#    # 2. 物理損失（連続の式）
#    u = predicted_velocity[:, 0, :, :]  # x方向の速度
#    v = predicted_velocity[:, 1, :, :]  # y方向の速度
#
#    # 中心差分による勾配計算
#    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
#    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)
#
#    # 内部の点のみを考慮
#    du_dx = du_dx[:, 1:-1, :]
#    dv_dy = dv_dy[:, :, 1:-1]
#
#    # ダイバージェンス（連続の式）
#    divergence = du_dx + dv_dy
#
#    # 物理損失
#    physics_loss = torch.mean(divergence ** 2)
#
#    # 3. 境界条件による損失
#    # a. 壁沿いの速度は0
#    wall_mask = torch.ones_like(u, dtype=torch.bool)
#
#    # x座標をインデックスに変換する関数(メッシュの区切り幅でインデックスは変わる)
#    def x_to_index(x):
#        return int(round(x / dx))
#
#    # 入り口と出口のxインデックス範囲
#    inlet_x_start = x_to_index(1.0)   # x=1.0
#    inlet_x_end = x_to_index(2.2)     # x=2.2
#    outlet_x_start = x_to_index(1.4)  # x=1.4
#    outlet_x_end = x_to_index(1.8)    # x=1.8
#
#    # 入り口と出口を壁マスクから除外
#    wall_mask[:, 0, inlet_x_start:inlet_x_end] = False   # y=0（入り口）
#    wall_mask[:, -1, outlet_x_start:outlet_x_end] = False  # y=31（出口）
#
#    # 壁での速度
#    u_wall = u[wall_mask]
#    v_wall = v[wall_mask]
#
#    # 壁の損失
#    wall_loss = torch.mean(u_wall ** 2 + v_wall ** 2)
#
#    # b. 入り口の風速は一定
#    inlet_mask = torch.zeros_like(u, dtype=torch.bool)
#    inlet_mask[:, 0, inlet_x_start:inlet_x_end] = True
#
#    # 入り口での予測速度
#    u_inlet_pred = u[inlet_mask]
#    v_inlet_pred = v[inlet_mask]
#
#    # 入り口の真の速度
#    u_inlet_true = inlet_wind_speed[:,0].repeat_interleave(inlet_x_end - inlet_x_start)
#    v_inlet_true = inlet_wind_speed[:,1].repeat_interleave(inlet_x_end - inlet_x_start)
#
#    # 入り口の損失
#    inlet_loss = torch.mean((u_inlet_pred - u_inlet_true) ** 2 + (v_inlet_pred - v_inlet_true) ** 2)
#
#    # 4. 総合損失
#    total_loss = (data_loss_weight * data_loss +
#                  physics_loss_weight * physics_loss +
#                  wall_loss_weight * wall_loss +
#                  inlet_loss_weight * inlet_loss)
#
#    # 個別の損失を辞書で返す
#    losses = {
#        'data_loss': data_loss,
#        'physics_loss': physics_loss,
#        'wall_loss': wall_loss,
#        'inlet_loss': inlet_loss
#    }
#
#    return total_loss, losses


def compute_losses(predicted_velocity, predicted_pressure, true_velocity, true_pressure, inlet_wind_speed, fluid_density= 1.225 , fluid_viscosity=1.48e-5):
    """
    推定結果からデータ損失、物理損失、壁の損失、入り口の損失を計算し、総合損失を返す関数。

    Parameters:
    - predicted_velocity: モデルが予測した速度場（バッチサイズ, 2, 32, 32）
    - predicted_pressure: モデルが予測した圧力場（バッチサイズ, 1, 32, 32）
    - true_velocity: 正解の速度場（バッチサイズ, 2, 32, 32）
    - true_pressure: 正解の圧力場（バッチサイズ, 1, 32, 32）
    - inlet_wind_speed: 入り口風速の定数値（バッチサイズ, 2）
    - fluid_density: 流体の密度（スカラー）
    - fluid_viscosity: 流体の動粘性係数（スカラー）

    Returns:
    - total_loss: 総合損失
    - losses: 個別の損失を含む辞書
    """
    # 重みの設定
    data_loss_weight = 5.0
    pressure_data_loss_weight = 5.0
    continuity_loss_weight = 0.1
    momentum_loss_weight = 0.1
    wall_loss_weight = 0.5
    inlet_loss_weight = 1.0

    # メッシュの解像度
    nx, ny = 32, 32
    dx = dy = 3.2 / 32  # セルサイズ（0.1m）

    # 1. データ損失（MSE）
    data_loss = torch.mean((predicted_velocity - true_velocity) ** 2)

    # 圧力のデータ損失（MSE）
    pressure_data_loss = torch.mean((predicted_pressure - true_pressure) ** 2)

    # 2. 物理損失
    # a. 連続の式（質量保存則）
    u = predicted_velocity[:, 0, :, :]  # x方向の速度 (バッチサイズ, H, W)
    v = predicted_velocity[:, 1, :, :]  # y方向の速度 (バッチサイズ, H, W)
    p = predicted_pressure[:, 0, :, :]  # 圧力場 (バッチサイズ, H, W)

    # パディングして境界条件を考慮
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode='replicate')  # (バッチサイズ, H+2, W+2)
    v_padded = torch.nn.functional.pad(v, (1, 1, 1, 1), mode='replicate')
    p_padded = torch.nn.functional.pad(p, (1, 1, 1, 1), mode='replicate')

    # 中心差分による勾配計算
    du_dx_full = (u_padded[:, 1:-1, 2:] - u_padded[:, 1:-1, :-2]) / (2 * dx)
    du_dx = du_dx_full[:, 1:-1, 1:-1]  # (バッチサイズ, H-2, W-2)

    du_dy_full = (u_padded[:, 2:, 1:-1] - u_padded[:, :-2, 1:-1]) / (2 * dy)
    du_dy = du_dy_full[:, 1:-1, 1:-1]

    dv_dx_full = (v_padded[:, 1:-1, 2:] - v_padded[:, 1:-1, :-2]) / (2 * dx)
    dv_dx = dv_dx_full[:, 1:-1, 1:-1]

    dv_dy_full = (v_padded[:, 2:, 1:-1] - v_padded[:, :-2, 1:-1]) / (2 * dy)
    dv_dy = dv_dy_full[:, 1:-1, 1:-1]

    # 連続の式（ダイバージェンス）
    continuity = du_dx_full + dv_dy_full  # (バッチサイズ, H, W)
    continuity = continuity[:, 1:-1, 1:-1]  # (バッチサイズ, H-2, W-2)

    # 連続の式の損失
    continuity_loss = torch.mean(continuity ** 2)

    # b. ナビエ–ストークス方程式（運動量保存則）
    # 圧力勾配（中心差分）
    dp_dx_full = (p_padded[:, 1:-1, 2:] - p_padded[:, 1:-1, :-2]) / (2 * dx)
    dp_dx = dp_dx_full[:, 1:-1, 1:-1]

    dp_dy_full = (p_padded[:, 2:, 1:-1] - p_padded[:, :-2, 1:-1]) / (2 * dy)
    dp_dy = dp_dy_full[:, 1:-1, 1:-1]

    # 速度の2階微分（粘性項）
    d2u_dx2_full = (u_padded[:, 1:-1, 2:] - 2 * u_padded[:, 1:-1, 1:-1] + u_padded[:, 1:-1, :-2]) / (dx ** 2)
    d2u_dx2 = d2u_dx2_full[:, 1:-1, 1:-1]

    d2u_dy2_full = (u_padded[:, 2:, 1:-1] - 2 * u_padded[:, 1:-1, 1:-1] + u_padded[:, :-2, 1:-1]) / (dy ** 2)
    d2u_dy2 = d2u_dy2_full[:, 1:-1, 1:-1]

    d2v_dx2_full = (v_padded[:, 1:-1, 2:] - 2 * v_padded[:, 1:-1, 1:-1] + v_padded[:, 1:-1, :-2]) / (dx ** 2)
    d2v_dx2 = d2v_dx2_full[:, 1:-1, 1:-1]

    d2v_dy2_full = (v_padded[:, 2:, 1:-1] - 2 * v_padded[:, 1:-1, 1:-1] + v_padded[:, :-2, 1:-1]) / (dy ** 2)
    d2v_dy2 = d2v_dy2_full[:, 1:-1, 1:-1]

    # ラプラシアン
    laplacian_u = d2u_dx2 + d2u_dy2
    laplacian_v = d2v_dx2 + d2v_dy2

    # 中央部分の速度と圧力
    u_center = u[:, 1:-1, 1:-1]  # (バッチサイズ, H-2, W-2)
    v_center = v[:, 1:-1, 1:-1]
    p_center = p[:, 1:-1, 1:-1]

    # 慣性項
    inertia_u = u_center * du_dx + v_center * du_dy
    inertia_v = u_center * dv_dx + v_center * dv_dy

    # 運動量方程式の残差
    momentum_u = fluid_density * inertia_u + dp_dx - fluid_viscosity * laplacian_u
    momentum_v = fluid_density * inertia_v + dp_dy - fluid_viscosity * laplacian_v

    # 運動量保存則の損失
    momentum_loss = torch.mean(momentum_u ** 2 + momentum_v ** 2)

    # 3. 境界条件による損失
    # a. 壁沿いの速度は0
    wall_mask = torch.ones_like(u, dtype=torch.bool)

    # x座標をインデックスに変換する関数
    def x_to_index(x):
        return int(round(x / dx))

    # 入り口と出口のxインデックス範囲
    inlet_x_start = x_to_index(1.0)   # x=1.0
    inlet_x_end = x_to_index(2.2)     # x=2.2
    outlet_x_start = x_to_index(1.4)  # x=1.4
    outlet_x_end = x_to_index(1.8)    # x=1.8

    # 入り口と出口を壁マスクから除外
    wall_mask[:, 0, inlet_x_start:inlet_x_end] = False   # y=0（入り口）
    wall_mask[:, -1, outlet_x_start:outlet_x_end] = False  # y=31（出口）

    # 壁での速度
    u_wall = u[wall_mask]
    v_wall = v[wall_mask]

    # 壁の損失
    wall_loss = torch.mean(u_wall ** 2 + v_wall ** 2)

    # b. 入り口の風速は一定
    inlet_mask = torch.zeros_like(u, dtype=torch.bool)
    inlet_mask[:, 0, inlet_x_start:inlet_x_end] = True

    # 入り口での予測速度
    u_inlet_pred = u[inlet_mask]
    v_inlet_pred = v[inlet_mask]

    # 入り口の真の速度
    u_inlet_true = inlet_wind_speed[:, 0].unsqueeze(1).repeat(1, inlet_x_end - inlet_x_start).flatten()
    v_inlet_true = inlet_wind_speed[:, 1].unsqueeze(1).repeat(1, inlet_x_end - inlet_x_start).flatten()

    # 入り口の損失
    inlet_loss = torch.mean((u_inlet_pred - u_inlet_true) ** 2 + (v_inlet_pred - v_inlet_true) ** 2)

    # 4. 総合損失
    total_loss = (
        data_loss_weight * data_loss +
        pressure_data_loss_weight * pressure_data_loss +
        continuity_loss_weight * continuity_loss +
        momentum_loss_weight * momentum_loss +
        wall_loss_weight * wall_loss +
        inlet_loss_weight * inlet_loss
    )

    # 個別の損失を辞書で返す
    losses = {
        'data_loss': data_loss,
        'pressure_data_loss': pressure_data_loss,
        'continuity_loss': continuity_loss,
        'momentum_loss': momentum_loss,
        'wall_loss': wall_loss,
        'inlet_loss': inlet_loss
    }

    return total_loss, losses

