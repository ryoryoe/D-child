#diffusion modelを使った学習を行うためのコード
import os
import math
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
# ハイパーパラメータをdataclassとjsonを使って管理するため
import json
from dataclasses import dataclass, field
import pandas as pd
# 漢字の画像を生成するため
from fontTools import ttLib
from PIL import Image, ImageFont, ImageDraw
from mymodule import Preprocessing, file_maker
from preprocess import Preprocessing_standard
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
# UNetはこちらを利用しています
from modules import UNet
from matplotlib.ticker import MultipleLocator

# %%==========================================================================
# Denoising Diffusion Probabilistic Models
# ============================================================================
class DDPM(nn.Module):
    def __init__(self, T, device):
        super().__init__()
        self.device = device
        self.T = T #ノイズを加える回数
        # β1 and βt はオリジナルの ddpm reportに記載されている値を採用します
        self.beta_1 = 1e-4 #t=1のノイズの大きさ
        self.beta_T = 0.02 #t=Tのノイズの大きさ
        # β = [β1, β2, β3, ... βT] (length = T)
        self.betas = torch.linspace(self.beta_1, self.beta_T, T, device=device)#t=1からt=Tまでのノイズの大きさを線形に変化させる
        # α = [α1, α2, α3, ... αT] (length = T)
        self.alphas = 1.0 - self.betas #最初の位置から今の位置までに加えるノイズの合計
        # α bar [α_bar_1, α_bar_2, ... , α_bar_T] (length = T)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) #αの配列

    def diffusion_process(self, x0, t=None):
        if t is None:
            t = torch.randint(low=1, high=self.T, size=(x0.shape[0],), device=self.device) #最初に受け取る値はnoneで、その場合はランダムにtを選ぶ
        noise = torch.randn_like(x0, device=self.device) #ノイズを生成
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ

def denoising_process(self, model, img, ts):#与えられた画像からノイズを取り除く関数
    batch_size = img.shape[0] #バッチサイズを取得
    model.eval() #モデルを評価モードにする
    with torch.no_grad():
        time_step_bar = tqdm(reversed(range(1, ts)), leave=False, position=0) #ノイズを復元するために逆順にループ
        for t in time_step_bar:  # ts, ts-1, .... 3, 2, 1
            # 整数値のtをテンソルに変換。テンソルのサイズは(バッチサイズ, )
            time_tensor = (torch.ones(batch_size, device=self.device) * t).long()#tの値をバッチサイズ分用意
            # 現在の画像からノイズを予測
            prediction_noise = model(img, time_tensor)#ノイズを予測
            # 現在の画像からノイズを少し取り除く
            img = self._calc_denoising_one_step(img, time_tensor, prediction_noise)
    model.train()
    # 0~255のデータに変換して返す
    img = img.clamp(-1, 1)
    img = (img + 1) / 2
    img = (img * 255).type(torch.uint8)
    return img

def _calc_denoising_one_step(self, img, time_tensor, prediction_noise): #ノイズを計算する関数
        beta = self.betas[time_tensor].reshape(-1, 1, 1, 1) #tの値に応じてノイズの大きさを選ぶ
        sqrt_alpha = torch.sqrt(self.alphas[time_tensor].reshape(-1, 1, 1, 1)) #tの値に応じてαの値を選ぶ
        alpha_bar = self.alpha_bars[time_tensor].reshape(-1, 1, 1, 1) #tの値に応じてα_barの値を選ぶ
        sigma_t = torch.sqrt(beta)
        noise = torch.randn_like(img, device=self.device) if time_tensor[0].item() > 1 else torch.zeros_like(img, device=self.device) #t=1の時はノイズを0にする.   
        img = 1 / sqrt_alpha * (img - (beta / (torch.sqrt(1 - alpha_bar))) * prediction_noise) + sigma_t * noise #ノイズを取り除く
        return img
def sort_and_combine_strings(input_array):
    # 配列 a と b を初期化
    a = []
    b = []

    # 各要素を確認して振り分け
    for string in input_array:
        if not '-' in string:
            a.append(string)
        if '-' in string:
            b.append(string)

    # 配列 a と b を結合
    file_names = a + b
    return file_names
# %%==========================================================================
# ddpm training
# ============================================================================
def ddpm_train(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")
    # 必要なモデルなどを生成
    train_path = params.train_path
    train_eval_path = params.train_eval_path
    test_path = params.test_path
    test_eval_path = params.test_eval_path
    
    estimate_path = f"{test_path}"
    estimate_eval_path = f"{test_eval_path}"
    
    if params.learning ==1: 
        print("train_data_preprocessing_start")
        train_x,train_y,file_names,_,_ = Preprocessing_standard(train_path,train_eval_path,params.width,params.standard,params.cut)
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        #train_dara_ver10でnanを出さないための応急処置
        if str(10) in params.train_file_path:
             a = train_x[:3350]
             b = train_x[:3400]
             train_x = torch.cat([a,b])
             a = train_y[:3350]
             b = train_y[:3400]
             train_y = torch.cat([a,b])
        train_x, train_y = train_x[:params.cut_size], train_y[:params.cut_size]
        trainset = torch.utils.data.TensorDataset(train_x,train_y)
        # データセットのサイズを計算
        dataset_size = len(trainset)
        test_size = int(dataset_size * params.rate)  # データセットの10%をテストデータとして使用
        train_size = dataset_size - test_size  # 残りを訓練データとして使用
        
        # データセットをランダムに分割
        train_dataset, test_dataset = random_split(trainset, [train_size, test_size])
        
        dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = params.batch_size, num_workers = 2, drop_last=True)
        testloader = torch.utils.data.DataLoader(test_dataset,batch_size = params.batch_size, num_workers = 2, drop_last=True)
    eval_x,eval_y,file_names_estimate,avg_list,std_list = Preprocessing_standard(estimate_path,estimate_eval_path,params.width,params.standard,params.cut)
    eval_x = torch.tensor(eval_x, dtype=torch.float32)
    eval_y = torch.tensor(eval_y, dtype=torch.float32)
    
    evalset = torch.utils.data.TensorDataset(eval_x,eval_y)
    estimate_loader= torch.utils.data.DataLoader(evalset,batch_size = 1, num_workers = 2, drop_last=True)
    ddpm = DDPM(params.time_steps, device)
    if params.learning == 1:
        file_maker(f"../result/{params.output_path}")
        model = UNet(params.image_ch, params.image_ch).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
        loss_fn = torch.nn.MSELoss()

        start_epoch = 1
        loss_list = []
        loss_list_test = []
        # training
        epoch_bar = tqdm(range(start_epoch, params.epochs+1))
        for epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch:{epoch}")
            loss_tmp = 0
            iter_bar = tqdm(dataloader, leave=False)
            train_loss = 0
            test_loss = 0
            avg_test_loss = 0
            model.train()
            loss_count=0
            for iter, (x,y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                # xにノイズを加えて学習データを作成する
                xt, t, noise = ddpm.diffusion_process(x)
                # モデルによる予測〜誤差逆伝播
                out = model(xt, t)#modelに一度通せばノイズを取り除いた画像が出てくるので正解との誤差を計算できる
                loss = loss_fn(y, out)#ノイズと予測したノイズの誤差を計算
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_bar.set_postfix({"loss=": f"{loss.item():.2e}"})

            # テストデータでの損失を計算
            model.eval()
            for test_counter,(inputs,labels) in enumerate(testloader):
                inputs,labels = inputs.to(device),labels.to(device)
                optimizer.zero_grad()
                xt, t, noise = ddpm.diffusion_process(inputs)
                out = model(xt, t)
                loss = loss_fn(labels,out)
                test_loss += loss.item()
            avg_train_loss = train_loss/len(train_dataset)
            avg_test_loss = test_loss/len(test_dataset)
            loss_list.append(avg_train_loss)
            loss_list_test.append(avg_test_loss)
            epoch_bar.set_postfix({"train_loss": f"{avg_train_loss:.2e}", "test_loss": f"{avg_test_loss:.2e}"})

        fig=plt.figure()
        plt.plot(loss_list,label='valid', lw=2, c='b')
        plt.plot(loss_list_test,label='test', lw=2, c='k')
        plt.grid()
        plt.rcParams["font.size"] = 10
        plt.xlabel("Epoch")
        plt.ylabel("Loss function")
        plt.legend()
        ax = plt.gca()  # 現在の軸を取得
        ax.xaxis.set_major_locator(MultipleLocator(params.epochs*0.1)) 
        plt.savefig(f"../result/{params.output_path}/loss_{params.output_path}.pdf")
        
        torch.save(model,f"../result/{params.output_path}/weight_{params.output_path}.pth")
    print("estimate_start")
    save_path = params.file_path
    model = torch.load(params.weight_eval_path)
    model = model.to(device)
    eval_loss = 0
    loss_list = []
    criterion = nn.MSELoss()
    count = 0
    output_list = []
    model.eval()
    avg_list = np.reshape(avg_list,[-1,1])
    std_list = np.reshape(std_list,[-1,1])
    for counter,(data,evaly) in enumerate(estimate_loader):
            data = data.to(device)
            evaly = evaly.to(device)
            #torch型でt=1000を定義(型はint)
            t = torch.ones(data.shape[0],dtype=torch.int32,device=device)*params.time_steps
            xt, t, noise = ddpm.diffusion_process(data)
            p = model(xt,t)
            p_output = p.detach().cpu().numpy()
            if params.standard == 1:
                p_output = np.reshape(p_output,[params.batch_size,-1])
                #standardization
                avg = avg_list[counter*params.batch_size:params.batch_size*(counter+1)]
                std = std_list[counter*params.batch_size:params.batch_size*(counter+1)]
                p_output *= std
                p_output += avg
                p_output = np.reshape(p_output,[params.batch_size,params.width,params.width,2])
            output_list.append(p_output)
            count+=1
            if count > params.end_estimate_number:
                break
    output_list = np.array(output_list)
    output_list = np.reshape(output_list,[count,2,-1])
    print(output_list.shape)
    for i in tqdm(range(len(output_list)),total=len(output_list)):
        out = output_list[i]
        out = out.T
        file_maker(f"../result/{params.output_path}")
        file_maker(f"../result/{params.output_path}/{params.file_path}")
        pd.DataFrame(out,columns=["X Velocity","Y Velocity"]).to_csv(f"../result/{params.output_path}/{params.file_path}/estimate_{file_names_estimate[i]}.csv", index=False)
        


@dataclass
class HyperParameters:
    #ファイル関連
    task_name: str = "estimate_velocity"
    #output_path: str = "diffusion_model_0221_T=20_from5_to20"
    output_path: str = "diffusion_model_0222_T=20_cut_under0.5" #出力先のフォルダ名
    file_path: str = "train_data_ver6_test" #推定に使うデータのフォルダ
    train_file_path = "train_data_ver10" #学習データのフォルダ
    train_path: str = f"../{train_file_path}/Time=5" #学習データ
    train_eval_path: str  = f"../{train_file_path}/Time=20" #学習データの正解ラベル
    test_path: str = f"../{file_path}/Time=5" #推定に使うデータ
    test_eval_path: str  = f"../{file_path}/Time=20" #推定に使うデータ(意味ない)
    weight_eval_path = f"../result/{output_path}/weight_{output_path}.pth" #学習済みモデルの名前
    
    #ハイパーパラメーター
    cut_size: int = 300000 #訓練データのサイズ(実際には10%はテストデータとして使う。全て使う時は大きい数を指定)
    learning = 1 #1で学習を行う,0で学習を行わずに推定のみを行う
    standard = 0 #1で標準化を行う,0で行わない
    epochs: int = 100 #エポック数
    width: int = 32 #画像の幅
    batch_size: int = 256 #バッチサイズ
    lr: float = 1.0e-3 #学習率
    time_steps: int =  1000  # T もう少し小さくても良いはず,何回ノイズを加えるか
    image_ch: int = 2 #画像のチャンネル数(xとyの速度の2つ)
    end_estimate_number=100 #推定するデータの数(多すぎる推定データを与えたときにこの数で推定をやめる)
    rate = 0.1 #訓練データとテストデータの割合(前処理が終わっているデータの何割をテストデータとして使うか)
    cut = 0.5 #cut以下の速度の値を0にする(学習を簡単にするために一定以下の速度を切り落とす,切り落とさない時は0を指定,0,5ぐらいで対象以外の部分を除ける)

params = HyperParameters()
ddpm_train(params)
