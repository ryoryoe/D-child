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
import json
from dataclasses import dataclass, field
import pandas as pd
from fontTools import ttLib
from PIL import Image, ImageFont, ImageDraw
from mymodule import Preprocessing, file_maker
from preprocess import Preprocessing_standard
from modules import UNet,Encoder,Decoder

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
    log_dir = make_save_dir_and_save_params(params)
    model_path = os.path.join(log_dir, f"model_weight_on_{device}")
    # 必要なモデルなどを生成
    train_path = params.train_path
    train_eval_path = params.train_eval_path
    test_path = params.test_path
    test_eval_path = params.test_eval_path
    
    estimate_path = f"{test_path}"
    estimate_eval_path = f"{test_eval_path}"
    
    if params.learning ==1: 
        print("train_data_preprocessing_start")
        train_x,train_y,file_names,_,_ = Preprocessing_standard(train_path,train_eval_path,params.dmax,params.dmin,params.width)
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        trainset = torch.utils.data.TensorDataset(train_x,train_y)
        dataloader = torch.utils.data.DataLoader(trainset,batch_size = params.batch_size, shuffle = True, num_workers = 2, drop_last=True)
    eval_x,eval_y,file_names_estimate,avg_list,std_list = Preprocessing_standard(estimate_path,estimate_eval_path,params.dmax,params.dmin,params.width)
    eval_x = torch.tensor(eval_x, dtype=torch.float32)
    eval_y = torch.tensor(eval_y, dtype=torch.float32)
    
    evalset = torch.utils.data.TensorDataset(eval_x,eval_y)
    estimate_loader= torch.utils.data.DataLoader(evalset,batch_size = params.batch_size, num_workers = 2, drop_last=True)
    ddpm = DDPM(params.time_steps, device)
    model = UNet(params.image_ch, params.image_ch).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    if params.learning == 1:
        combined_params = list(encoder.parameters()) + list(model.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(combined_params, lr=params.lr)
        loss_fn = torch.nn.MSELoss()

        start_epoch = 1
        loss_logger = []
        loss_min = 9e+9
        # training
        model.train()
        epoch_bar = tqdm(range(start_epoch, params.epochs+1))
        for epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch:{epoch}")
            loss_tmp = 0
            iter_bar = tqdm(dataloader, leave=False)
            for iter, (x,y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                encode_avg,encode_std = encoder(x)
                x = torch.stack((encode_avg, encode_std), dim=1)
                xt, t, noise = ddpm.diffusion_process(x)
                out = model(xt, t)#modelに一度通せばノイズを取り除いた画像が出てくるので正解との誤差を計算できる
                out = decoder(out)
                loss = loss_fn(y, out)#ノイズと予測したノイズの誤差を計算
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # lossの経過記録
                iter_bar.set_postfix({"loss=": f"{loss.item():.2e}"})
                loss_tmp += loss.item()

            loss_logger.append(loss_tmp / (iter + 1))
            epoch_bar.set_postfix({"loss=": f"{loss_logger[-1]:.2e}"})
            # 保存処理
            # lossの経過グラフを出力
            save_loss_logger_and_graph(log_dir, loss_logger)
            # lossが最小の場合は重みデータを保存
            if loss_min >= loss_logger[-1]:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss_logger,
                            }, model_path)
                loss_min = loss_logger[-1]
        
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
    for counter,(data,_) in enumerate(estimate_loader):
            data = data.to(device)
            encode_avg,encode_std = encoder(data)
            x = torch.stack((encode_avg, encode_std), dim=1)
            xt, t, noise = ddpm.diffusion_process(x)
            out = model(xt, t)#modelに一度通せばノイズを取り除いた画像が出てくるので正解との誤差を計算できる
            out = decoder(out)
            p_output = out.detach().cpu().numpy()
            
            #standardization
            avg = avg_list[counter*params.batch_size:params.batch_size*(counter+1)]
            std = std_list[counter*params.batch_size:params.batch_size*(counter+1)]
            p_output *= std
            p_output += avg
            output_list.append(p_output)
            count+=params.batch_size
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
        
#何の関数かをメモしていく
def load_checkpoint(params, model, optimizer, model_path, device):
    print(f"load model {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss_logger = checkpoint["loss"]
    loss_min = min(loss_logger)
    print(start_epoch)
    return model, optimizer, start_epoch, loss_logger, loss_min


def make_save_dir_and_save_params(params):
    # タスクの保存フォルダ
    log_dir = os.path.join(r"../result", params.output_path, params.file_path,params.task_name)
    os.makedirs(log_dir, exist_ok=True)
    # epoch, iter毎のデータは多くなるので別フォルダを作る
    log_dir_hist = os.path.join(log_dir, "hist")
    os.makedirs(log_dir_hist, exist_ok=True)
    # 設定ファイルの保存
    with open(os.path.join(log_dir, "parameters.json"), 'w') as f:
        json.dump(vars(params), f, indent=4)
    return log_dir


def save_loss_logger_and_graph(log_dir, loss_logger):
    # loss履歴情報を保管しつつ、グラフにして画像としても書き出す
    torch.save(loss_logger, os.path.join(log_dir, "loss_logger.pt"))
    fig, ax = plt.subplots(1,1)
    epoch = range(len(loss_logger))
    ax.plot(epoch, loss_logger, label="train_loss")
    ax.set_ylim(0, loss_logger[-1]*5)
    ax.legend()
    fig.savefig(os.path.join(log_dir, "loss_history.jpg"))
    plt.clf()
    plt.close()




@dataclass
class HyperParameters:
    task_name: str = "estimate_velocity"
    output_path: str = "vae_diffusion_model_0127"
    file_path: str = "train_data_ver6"
    train_path: str = "../train_data_ver6/Time=5"
    train_eval_path: str  = "../train_data_ver6/Time=20"
    test_path: str = f"../{file_path}/Time=5"
    test_eval_path: str  = f"../{file_path}/Time=20"
    weight_eval_path = f"../result/{output_path}/weight_{output_path}.pth"
    learning = 0
    epochs: int = 1000
    width: int = 32
    img_save_steps: int = 100
    batch_size: int = 1
    lr: float = 1.0e-3
    time_steps: int = 1000  # T もう少し小さくても良いはず,何回ノイズを加えるか
    load_file: bool = True
    pix: int = 32
    font_file: str = r"./ヒラギノ角ゴシック W5.ttc"
    image_ch: int = 2
    dmax = 100
    dmin = 0
    end_estimate_number=100
params = HyperParameters()
ddpm_train(params)
