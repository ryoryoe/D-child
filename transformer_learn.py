#複数の部屋の流れを学習するプログラム
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
from dataclasses import dataclass, field
import pandas as pd
from mymodule import Preprocessing, file_maker,condition_text
from preprocess import Preprocessing_standard_mix_for, Preprocessing_high_resolution
from torch.utils.data import TensorDataset, DataLoader
from modules import Input_Mix_divide_skip_detailed,Input_Mix_divide_skip_detailed_12
import modules
from modules_ver2_0429 import  UNet as UNet2
from matplotlib.ticker import MultipleLocator
from transformer_test import CFDTransformerModel
from transformer_preprocess import CFDVelocityDataset 


def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):#重みを保存
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
 
def load_checkpoint(model, optimizer, filename='checkpoint.pth'): #重みを読み込む
    print(f"load {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def ddpm_train(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")
    # 必要なモデルなどを生成
    
    
    if params.learning ==1: 
        print("train_data_preprocessing_start")
        #trainデータの前処理
        train_dataset = CFDVelocityDataset(params.train_directory, chunk_size=5000, max_value=1e3, test=params.test)
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

        #testデータの前処理
        test_dataset = CFDVelocityDataset(params.test_directory, chunk_size=500, max_value=1e3, test=True)
        test_dataloader = DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=True)

    estimate_dataset = CFDVelocityDataset(params.estimate_path, chunk_size=500, max_value=1e3, test=True)
    estimate_loader = DataLoader(estimate_dataset, batch_size=params.test_batch_size, shuffle=True)
    if params.learning == 1:
        file_maker(f"../result/{params.output_path}")
        condition_text(params.message,params.output_path)
        UNet_ = UNet2()

        #modelの定義
        model = CFDTransformerModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
        loss_fn = torch.nn.MSELoss()
        if params.additional_learning: 
            model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, f"../result/{params.output_path}/save_temp_weight/checkpoint_{params.output_path}_epoch={params.additional_epoch-1}.pth")
        else:
            file_maker(f"../result/{params.output_path}")
            condition_text(params.message,params.output_path)
            start_epoch = 0
        loss_list = []
        loss_list_test = []

        # training
        epoch_bar = tqdm(range(start_epoch, params.epochs+1))
        for epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch:{epoch}")
            iter_bar = tqdm(train_dataloader, leave=False)
            train_loss = 0
            test_loss = 0
            avg_test_loss = 0
            model.train()
            #for iter, (x,y,vx,vy,inlet_l,inlet_r,outlet_l,outlet_r) in enumerate(dataloader):
            for inputs,labels,_ in train_dataloader:
                x = inputs.to(device)
                y = labels.to(device)
                out,_ = model(x)

                loss = loss_fn(y, out)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_bar.set_postfix({"loss=": f"{loss.item():.2e}"})
            # テストデータでの損失を計算
            model.eval()
            with torch.no_grad():  # テスト時は勾配計算を行わない
                for inputs,labels,_ in test_dataloader:
                    x = inputs.to(device)
                    y = labels.to(device)
                    out,_ = model(x)
                    loss = loss_fn(y, out)
                    test_loss += loss.item()
                    iter_bar.set_postfix({"loss=": f"{loss.item():.2e}"})

            # 損失の平均を計算
            avg_train_loss = train_loss/(len(train_dataset)//params.batch_size)
            avg_test_loss = test_loss/(len(test_dataset)//params.test_batch_size)
            loss_list.append(avg_train_loss)
            loss_list_test.append(avg_test_loss)
            epoch_bar.set_postfix({"train_loss": f"{avg_train_loss:.2e}", "test_loss": f"{avg_test_loss:.2e}"})
            if epoch % params.save_interval == 0 and epoch != 0:    
                file_maker(f"../result/{params.output_path}/save_temp_weight")
                file_maker(f"../result/{params.output_path}/loss_file")
                save_checkpoint(model, optimizer, epoch, loss, f"../result/{params.output_path}/save_temp_weight/checkpoint_{params.output_path}_epoch={epoch}.pth")
                torch.save(model,f"../result/{params.output_path}/weight_{params.output_path}_epoch={epoch}.pth")
        torch.save(model,f"../result/{params.output_path}/weight_{params.output_path}.pth")
        fig=plt.figure()
        plt.plot(loss_list,label='valid', lw=2, c='b')
        plt.plot(loss_list_test,label='test', lw=2, c='k')
        plt.grid()
        plt.rcParams["font.size"] = 10
        plt.xlabel("Epoch")
        plt.ylabel("Loss function")
        plt.legend()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(params.epochs*0.1)) 
        plt.savefig(f"../result/{params.output_path}/loss_{params.output_path}.pdf") 
        if epoch >= 10:
            fig=plt.figure()
            plt.plot(loss_list[10:],label='valid', lw=2, c='b')
            plt.plot(loss_list_test[10:],label='test', lw=2, c='k')
            plt.grid()
            plt.rcParams["font.size"] = 10
            plt.xlabel("Epoch")
            plt.ylabel("Loss function")
            plt.legend()
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(params.epochs*0.1)) 
            plt.savefig(f"../result/{params.output_path}/loss_{params.output_path}_from_100.pdf") 

    #evaluation
    print("estimate_start")
    save_path = params.file_path
    if params.byepoch:
        model = torch.load(params.weight_eval_path_byepoch)
    else:
        model = torch.load(params.weight_eval_path)
    model = model.to(device)
    eval_loss = 0
    loss_list = []
    criterion = nn.MSELoss()
    output_list = []
    file_names_estimate = []
    model.eval()
    for inputs,labels,file_names in estimate_loader:
            v = inputs.to(device)
            p,_ = model(v)
            p_output = p.detach().cpu().numpy()
            #output_list.append(p_output)
             # 初回はそのままリストに格納、それ以降は結合
            if len(output_list) == 0:
                output_list = p_output  # 初回はそのまま格納
                file_names_estimate = file_names
            else:
                # NumPy の concatenate を使用して結合
                output_list = np.concatenate((output_list, p_output), axis=0)
                file_names_estimate = np.concatenate((file_names_estimate, file_names), axis=0)
    output_list = np.reshape(output_list,[-1,2,32*32])
    for i in tqdm(range(len(output_list)),total=len(output_list)):
        out = output_list[i]
        out = out.T
        file_maker(f"../result/{params.output_path}")
        file_maker(f"../result/{params.output_path}/{params.file_path}")
        if params.byepoch:
            #file_maketint(f"../result/{params.output_path}/{params.file_path_byepoch}")
            pd.DataFrame(out,columns=["X Velocity","Y Velocity"]).to_csv(f"../result/{params.output_path}/{params.file_path_byepoch}/estimate_{file_names_estimate[i]}", index=False)
        else:
            pd.DataFrame(out,columns=["X Velocity","Y Velocity"]).to_csv(f"../result/{params.output_path}/{params.file_path}/estimate_{file_names_estimate[i]}", index=False)


@dataclass
class HyperParameters:
    #ファイル関連
    output_path: str = "inlet_value1_1224_transformer"
    message: str = "1~10秒目から11秒目を推定"
    file_path: str = "train_data_ver11" #推定に使うデータのフォルダ
    weight_eval_path = f"../result/{output_path}/weight_{output_path}.pth" #学習済みモデルの名前
    train_directory: str = f"/mnt/data1/tony/{file_path}"
    test_directory: str = f"/mnt/data1/tony/{file_path}"
    estimate_path: str = f"/mnt/data1/tony/{file_path}"
    
    #ハイパーパラメーター
    save_interval: int = 25 #何エポックごとにモデルを保存するか
    learning = 1 #1で学習を行う,0で学習を行わずに推定のみを行う
    standard = 0 #1で標準化を行う,0で行わない
    epochs: int = 2000 #エポック数
 #画像の幅
    #batch_size: int = 1 #バッチサイズ
    batch_size: int = 2048 #バッチサイズ
    test_batch_size: int = 16
    lr: float = 1.0e-3 #学習率
    time_steps: int =  1000  # T もう少し小さくても良いはず,何回ノイズを加えるか
    end_estimate_number=10000 #推定するデータの数(多すぎる推定データを与えたときにこの数で推定をやめる)
    rate = 0.1 #訓練データとテストデータの割合(前処理が終わっているデータの何割をテストデータとして使うか)
    cut = 0.5 #cut以下の速度の値を0にする(学習を簡単にするために一定以下の速度を切り落とす,切り落とさない時は0を指定,0,5ぐらいで対象以外の部分を除ける)

    byepoch = False #学習途中のファイルで推定するならTrue
    #target_epoch: int = 350 #どのエポックのモデルを使って推定するか
    target_epoch: int = 500 #どのエポックのモデルを使って推定するか
    weight_eval_path_byepoch = f"../result/{output_path}/weight_{output_path}_epoch={target_epoch}.pth" #学習済みモデルの名前
    file_path_byepoch: str = f"{file_path}_epoch_{target_epoch}_pattarn1_train" #推定に使うデータのフォルダ

    additional_learning = False #Trueですでに保存されている重みを読み込んで学習を再開する
    additional_epoch = 301 #学習を再開するエポック数(重みファイルのepoch+1)
    
    test = False #コード全体が動くかどうかをテストするときのフラグ
     
params = HyperParameters()
ddpm_train(params)
