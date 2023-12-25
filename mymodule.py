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

batch = 64
#--------------------------------------------------
#function
#--------------------------------------------------

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
    """if np.isnan(df).any():
        print(path)
        nan_indices = np.where(np.isnan(df))[0]
        print(nan_indices)
        sys.exit()"""
    #df = np.reshape(df, [-1,2])
    df_eval = pd.read_csv(eval_path)
    df_eval = df_eval.values
    df_eval = df_eval.astype(float)
    """if np.isnan(df_eval).any():
        print(f"eval{eval_path}")
        nan_indices = np.where(np.isnan(df_eval))[0]
        print(nan_indices)
        sys.exit()"""
    #df_eval = np.reshape(df_eval, [-1,2])
    """avg = np.mean(df)
    std = np.std(df)
    df -= avg
    df /= std"""
    return df,df_eval

# 並列処理を実装するためのPreprocessing関数
def Preprocessing(inputname,input_evalname, dmax, dmin):
    input_path = sorted(glob.glob(inputname + "/*.csv"), key=natural_keys)
    eval_path = sorted(glob.glob(input_evalname + "/*.csv"), key=natural_keys)
    num_cores = os.cpu_count()
    file_names = file_name_maker(f"{inputname}")
    # ProcessPoolExecutorを使用して各ファイルを並列処理
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_file, input_path,eval_path, [dmax] * len(input_path), [dmin] * len(input_path)),total=len(input_path)))
    # 結果を統合
    train,evals = zip(*results)
    train = np.asarray(train, dtype=float)
    evals = np.asarray(evals, dtype=float)
    """train = []
    evals = []
    for index in range(len(input_path)):
        df,df_eval = process_file(input_path[index],eval_path[index],0,0)
        train.append(df)
        evals.append(df_eval)
    train = np.asarray(train)
    evals = np.asarray(evals)"""
    # 最終的なデータセットの形状を調整
    train = np.reshape(train, [len(input_path),10000,2])
    evals = np.reshape(evals, [len(input_path),10000,2])
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
    print(f"delete_index={indices_to_remove}")
    #print(f"after_train={len(train)}")
    #print(f"after_eval={len(evals)}") 
    train = np.reshape(train, [-1,100,100,2]).transpose(0,3,1, 2)
    evals = np.reshape(evals, [-1,100,100,2]).transpose(0,3,1,2)
    file_names = [item for idx, item in enumerate(file_names) if idx not in indices_to_remove]
    return train, evals,file_names

        

def atoi(text):
        return int(text) if text.isdigit() else text
def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#---------------------------------------------------

#class
class Simple2DCNN(nn.Module):
    def __init__(self):
        super(Simple2DCNN, self).__init__()
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

