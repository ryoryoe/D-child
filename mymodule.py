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
    with open(output, 'w') as file:
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

