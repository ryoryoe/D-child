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
from mymodule import Preprocessing, file_maker,condition_text
from preprocess import Preprocessing_standard
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
# UNetはこちらを利用しています
from modules import UNet,conditional_diffusion_0406,conditional_diffusion_0407_sum,conditional_diffusion_0407_sum_and_cat,Input_VModel,Input_VModel2_0504,Input_2VModel
from modules_ver2_0429 import  UNet as UNet2
from matplotlib.ticker import MultipleLocator

# %%==========================================================================
# 拡散モデルで回帰問題を解く最初のやり方でノイズを推定している
# ============================================================================

#vx_list = torch.tensor([-0.9,1.0,-0.6,-0.5,-1.2,-1.2,-0.5,-0.6,1.0,-0.9,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
#vy_list = torch.tensor([0.9,1.0,1.2,1.5,1.3,1.3,1.5,1.2,1.0,0.9,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
class DDPM(nn.Module):
    def __init__(self, T, device):
        super().__init__()
        self.device = device
        self.T = T #ノイズを加える回数
        # β1 and βt はオリジナルの ddpm reportに記載されている値を採用します
        self.beta_1 = 1e-6 #t=1のノイズの大きさ(最初1.0e-4)
        self.beta_T = 2.0e-4 #t=Tのノイズの大きさ(最初0.02)
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

    def denoising_process(self, model, img, ts,v):#与えられた画像からノイズを取り除く関数
        batch_size = img.shape[0] #バッチサイズを取得
        model.eval() #モデルを評価モードにする
        with torch.no_grad():
            time_step_bar = tqdm(reversed(range(1, ts)), leave=False, position=0) #ノイズを復元するために逆順にループ
            for t in time_step_bar:  # ts, ts-1, .... 3, 2, 1
                # 整数値のtをテンソルに変換。テンソルのサイズは(バッチサイズ, )
                time_tensor = (torch.ones(batch_size, device=self.device) * t).long()#tの値をバッチサイズ分用意
                # 現在の画像からノイズを予測
                #prediction_noise = model(img, time_tensor,v)#ノイズを予測
                prediction_noise = model(img, time_tensor)#ノイズを予測
                # 現在の画像からノイズを少し取り除く
                img = self._calc_denoising_one_step(img, time_tensor, prediction_noise)
        model.train()
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
        train_x,train_y,file_names,_,_,vx,vy,vx2,vy2 = Preprocessing_standard(train_path,train_eval_path,params.width,params.standard,params.cut,v2=True)
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        vx = torch.tensor(vx, dtype=torch.float32)
        vy = torch.tensor(vy, dtype=torch.float32)
        vx2 = torch.tensor(vx2, dtype=torch.float32)
        vy2 = torch.tensor(vy2, dtype=torch.float32)
        #train_dara_ver10でnanを出さないための応急処置
        if str(10) in params.train_file_path:
             a = train_x[:3350]
             b = train_x[:3400]
             train_x = torch.cat([a,b])
             a = train_y[:3350]
             b = train_y[:3400]
             train_y = torch.cat([a,b])
             a = vx[:3350]
             b = vx[:3400]
             vx = torch.cat([a,b])
             a = vy[:3350]
             b = vy[:3400]
             vy = torch.cat([a,b])
        train_x, train_y,vx,vy = train_x[:params.cut_size], train_y[:params.cut_size],vx[:params.cut_size],vy[:params.cut_size]
        vx2,vy2 = vx2[:params.cut_size],vy2[:params.cut_size]
        trainset = torch.utils.data.TensorDataset(train_x,train_y,vx,vy,vx2,vy2)
        #print(trainset.shape) 
        # データセットのサイズを計算
        #dataset_size = len(trainset)
        #test_size = int(dataset_size * params.rate)  # データセットの10%をテストデータとして使用
        #train_size = dataset_size - test_size  # 残りを訓練データとして使用
        
        # データセットをランダムに分割
        #train_dataset, test_dataset = random_split(trainset, [train_size, test_size])
        
        #dataloader = torch.utils.data.DataLoader(trainset,batch_size = params.batch_size, num_workers = 2, drop_last=True,shuffle=True)
        dataloader = torch.utils.data.DataLoader(trainset,batch_size = params.batch_size, num_workers = 2, drop_last=True)
        #testloader = torch.utils.data.DataLoader(test_dataset,batch_size = params.batch_size, num_workers = 2, drop_last=True)
    eval_x,eval_y,file_names_estimate,avg_list,std_list,vx_eval,vy_eval,vx2_eval,vy2_eval = Preprocessing_standard(estimate_path,estimate_eval_path,params.width,params.standard,params.cut,v2=True)
    eval_x = torch.tensor(eval_x, dtype=torch.float32)
    eval_y = torch.tensor(eval_y, dtype=torch.float32)
    vx_eval = torch.tensor(vx_eval, dtype=torch.float32)
    vy_eval = torch.tensor(vy_eval, dtype=torch.float32)
    vx2_eval = torch.tensor(vx2_eval, dtype=torch.float32)
    vy2_eval = torch.tensor(vy2_eval, dtype=torch.float32)
    evalset = torch.utils.data.TensorDataset(eval_x,eval_y,vx_eval,vy_eval,vx2_eval,vy2_eval)
    estimate_loader= torch.utils.data.DataLoader(evalset,batch_size = 1, num_workers = 2, drop_last=True)
    ddpm = DDPM(params.time_steps, device)
    if params.learning == 1:
        file_maker(f"../result/{params.output_path}")
        UNet_ = UNet2()
        #model = Input_VModel(UNet=UNet_).to(device)
        #model = Input_VModel2_0504(UNet=UNet_).to(device)
        model = Input_2VModel(UNet=UNet_).to(device)
        #model = UNet2().to(device)
        #model = conditional_diffusion_0406(params.image_ch, params.image_ch).to(device)
        #model = conditional_diffusion_0407_sum(params.image_ch, params.image_ch).to(device)
        #model = conditional_diffusion_0407_sum_and_cat(params.image_ch, params.image_ch).to(device)
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
            for iter, (x,y,vx,vy,vx2,vy2) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                vx,vy,vx2,vy2 = vx.to(device),vy.to(device),vx2.to(device),vy2.to(device)
                vx = vx.unsqueeze(1)
                vy = vy.unsqueeze(1)
                vx2,vy2 = vx2.unsqueeze(1),vy2.unsqueeze(1)
                v = torch.cat((vx,vy,vx2,vy2),dim=1)
                out = model(v)#modelに一度通せばノイズを取り除いた画像が出てくるので正解との誤差を計算できる
                loss = loss_fn(y, out)#ノイズと予測したノイズの誤差を計算
                #lossがinfなら終了
                if math.isinf(loss.item()):
                    print("lossがinfになりました")
                    print(f"{iter=}回目")
                    print(f"{vx=}")
                    print(f"{vy=}")
                    sys.exit()
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_bar.set_postfix({"loss=": f"{loss.item():.2e}"})

            # テストデータでの損失を計算
            """model.eval()
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
            loss_list_test.append(avg_test_loss)"""
            avg_train_loss = train_loss/(len(train_x)//params.batch_size)
            loss_list.append(avg_train_loss)
            epoch_bar.set_postfix({"train_loss": f"{avg_train_loss:.2e}"})
            if epoch % params.save_interval == 0:    
                torch.save(model,f"../result/{params.output_path}/weight_{params.output_path}_epoch={epoch}.pth")
        fig=plt.figure()
        plt.plot(loss_list,label='valid', lw=2, c='b')
        #plt.plot(loss_list_test,label='test', lw=2, c='k')
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
    if params.byepoch:
        model = torch.load(params.weight_eval_path_byepoch)
    else:
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
    for counter,(data,evaly,vx_eval,vy_eval,vx2_eval,vy2_eval) in enumerate(estimate_loader):
            vx_eval,vy_eval = vx_eval.to(device),vy_eval.to(device)
            vx_eval2,vy_eval2 = vx2_eval.to(device),vy2_eval.to(device)
            vx_eval = vx_eval.unsqueeze(1)
            vy_eval = vy_eval.unsqueeze(1)
            vx_eval2,vy_eval2 = vx_eval2.unsqueeze(1),vy_eval2.unsqueeze(1)
            v = torch.cat((vx_eval,vy_eval,vx_eval2,vy_eval2),dim=1)
            p = model(v)
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
        if params.byepoch:
            file_maker(f"../result/{params.output_path}/{params.file_path_byepoch}")
            pd.DataFrame(out,columns=["X Velocity","Y Velocity"]).to_csv(f"../result/{params.output_path}/{params.file_path_byepoch}/estimate_{file_names_estimate[i]}.csv", index=False)
        else:
            pd.DataFrame(out,columns=["X Velocity","Y Velocity"]).to_csv(f"../result/{params.output_path}/{params.file_path}/estimate_{file_names_estimate[i]}.csv", index=False)
    if params.learning == 1:
        condition_text(params.message,params.output_path)


@dataclass
class HyperParameters:
    #ファイル関連
    task_name: str = "estimate_velocity"
    output_path: str = "input_2v_0505" #出力先のフォルダ名
    #output_path: str = "input_v_0504_complicated_flow_to_20_add_relu_under_vy_0.8" #出力先のフォルダ名
    message: str = "入り口2つに別の速度を振っているcomp_ver2で学習" #学習内容
    file_path: str = "complicated_flow2_test" #推定に使うデータのフォルダ
    train_file_path = "complicated_flow2" #学習データのフォルダ
    train_path: str = f"../{train_file_path}/Time=20" #学習データ
    train_eval_path: str  = f"../{train_file_path}/Time=20" #学習データの正解ラベル
    test_path: str = f"../{file_path}/Time=20" #推定に使うデータ
    test_eval_path: str  = f"../{file_path}/Time=20" #推定に使うデータ(意味ない)
    weight_eval_path = f"../result/{output_path}/weight_{output_path}.pth" #学習済みモデルの名前
    
    #ハイパーパラメーター
    cut_size: int = 300000 #訓練データのサイズ(実際には10%はテストデータとして使う。全て使う時は大きい数を指定)
    save_interval: int = 50 #何エポックごとにモデルを保存するか
    learning = 0 #1で学習を行う,0で学習を行わずに推定のみを行う
    standard = 0 #1で標準化を行う,0で行わない
    epochs: int = 500 #エポック数
    width: int = 32 #画像の幅
    batch_size: int =256 #バッチサイズ
    lr: float = 1.0e-3 #学習率
    time_steps: int =  1000  # T もう少し小さくても良いはず,何回ノイズを加えるか
    image_ch: int = 2 #画像のチャンネル数(xとyの速度の2つ)
    end_estimate_number=100 #推定するデータの数(多すぎる推定データを与えたときにこの数で推定をやめる)
    rate = 0.1 #訓練データとテストデータの割合(前処理が終わっているデータの何割をテストデータとして使うか)
    cut = 0.5 #cut以下の速度の値を0にする(学習を簡単にするために一定以下の速度を切り落とす,切り落とさない時は0を指定,0,5ぐらいで対象以外の部分を除ける)

    byepoch = True #学習途中のファイルで推定するならTrue
    target_epoch: int = 50 #どのエポックのモデルを使って推定するか
    weight_eval_path_byepoch = f"../result/{output_path}/weight_{output_path}_epoch={target_epoch}.pth" #学習済みモデルの名前
    file_path_byepoch: str = f"{file_path}_epoch_{target_epoch}" #推定に使うデータのフォルダ
params = HyperParameters()
ddpm_train(params)