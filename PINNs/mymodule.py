import numpy as np
import csv
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
def condition_text(message, output,params):
    # ファイルを開き、messageの内容を書き込む
    with open(f"{params.root_path}/result/{output}/condition.txt", 'w') as file:
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


def atoi(text):
        return int(text) if text.isdigit() else text
def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#---------------------------------------------------

def compute_losses(predicted_velocity, true_velocity, inlet_wind_speed,loss_weights):
#def compute_losses(predicted_velocity, true_velocity, inlet_wind_speed):
    """
    推定結果からデータ損失、物理損失、壁の損失、入り口の損失を計算し、総合損失を返す関数。

    Parameters:
    - predicted_velocity: モデルが予測した速度場（バッチサイズ, 2, 32, 32）
    - true_velocity: 正解の速度場（バッチサイズ, 2, 32, 32）
    - inlet_wind_speed: 入り口風速の定数値（スカラー）

    Returns:
    - total_loss: 総合損失
    - losses: 個別の損失を含む辞書
    """
    data_loss_weight, physics_loss_weight, wall_loss_weight, inlet_loss_weight = loss_weights.get_weights()
    # 重みの設定
    #data_loss_weight = 10
    #physics_loss_weight = 0.1
    #wall_loss_weight = 0.5
    #inlet_loss_weight = 1.0
    
    #初期設定
    #data_loss_weight = 2.0
    #physics_loss_weight = 0.1
    #wall_loss_weight = 0.5
    #inlet_loss_weight = 0.5

    # メッシュの解像度
    nx, ny = 32, 32
    dx = dy = 3.2 / 32  # セルサイズ（0.1m）

    # 1. データ損失（MSE）
    data_loss = torch.mean((predicted_velocity - true_velocity) ** 2)

    # 2. 物理損失（連続の式）
    u = predicted_velocity[:, 0, :, :]  # x方向の速度
    v = predicted_velocity[:, 1, :, :]  # y方向の速度

    # 中心差分による勾配計算
    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)

    # 内部の点のみを考慮
    du_dx = du_dx[:, 1:-1, :]
    dv_dy = dv_dy[:, :, 1:-1]

    # ダイバージェンス（連続の式）
    divergence = du_dx + dv_dy

    # 物理損失
    physics_loss = torch.mean(divergence ** 2)

    # 3. 境界条件による損失
    # a. 壁沿いの速度は0
    wall_mask = torch.ones_like(u, dtype=torch.bool)

    # x座標をインデックスに変換する関数(メッシュの区切り幅でインデックスは変わる)
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
    u_inlet_true = inlet_wind_speed[:,0].repeat_interleave(inlet_x_end - inlet_x_start)
    v_inlet_true = inlet_wind_speed[:,1].repeat_interleave(inlet_x_end - inlet_x_start)

    # 入り口の損失
    inlet_loss = torch.mean((u_inlet_pred - u_inlet_true) ** 2 + (v_inlet_pred - v_inlet_true) ** 2)
    
    # 4. 重みパラメータのL2正則化
    l2_reg = (loss_weights.log_data_loss_weight ** 2 +
              loss_weights.log_physics_loss_weight ** 2 +
              loss_weights.log_wall_loss_weight ** 2 +
              loss_weights.log_inlet_loss_weight ** 2)

    # 正則化係数（ハイパーパラメータ）
    lambda_reg = 0.01  # 必要に応じて調整

    # 5. 総合損失
    total_loss = (data_loss_weight * data_loss +
                  physics_loss_weight * physics_loss +
                  wall_loss_weight * wall_loss +
                  inlet_loss_weight * inlet_loss +
                  lambda_reg * l2_reg)  # 正則化項を追加

    # 4. 総合損失
    #total_loss = (data_loss_weight * data_loss +
    #              physics_loss_weight * physics_loss +
    #              wall_loss_weight * wall_loss +
    #              inlet_loss_weight * inlet_loss)

    # 個別の損失を辞書で返す
    losses = {
        'data_loss': data_loss.detach().item(),
        'physics_loss': physics_loss.detach().item(),
        'wall_loss': wall_loss.detach().item(),
        'inlet_loss': inlet_loss.detach().item(),
    }

    return total_loss, losses

class LossWeights(nn.Module):
    def __init__(self):
        super(LossWeights, self).__init__()
        # 初期値を設定し、学習可能なパラメータとして定義
        # 対数スケールで定義し、重みが正の値になるようにします
        self.log_data_loss_weight = nn.Parameter(torch.tensor(0.0))
        self.log_physics_loss_weight = nn.Parameter(torch.tensor(0.0))
        self.log_wall_loss_weight = nn.Parameter(torch.tensor(0.0))
        self.log_inlet_loss_weight = nn.Parameter(torch.tensor(0.0))

    def get_weights(self):
        # 重みを正の値に変換
        data_loss_weight = torch.exp(self.log_data_loss_weight)
        physics_loss_weight = torch.exp(self.log_physics_loss_weight)
        wall_loss_weight = torch.exp(self.log_wall_loss_weight)
        inlet_loss_weight = torch.exp(self.log_inlet_loss_weight)
        return data_loss_weight, physics_loss_weight, wall_loss_weight, inlet_loss_weight
