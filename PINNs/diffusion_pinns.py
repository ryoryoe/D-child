#diffusion modelを使った学習を行うためのコード
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
from dataclasses import dataclass, field
import pandas as pd
from mymodule import file_maker,condition_text
from preprocess import Preprocessing_standard_mix_for, Preprocessing_high_resolution
from torch.utils.data import TensorDataset, DataLoader
from modules import Input_Mix_divide_skip_detailed,Input_Mix_divide_skip_detailed_12
import modules
import mymodule
from modules_ver2_0429 import  UNet as UNet2, UNet_high_resolution
from matplotlib.ticker import MultipleLocator

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
    train_path = params.train_path
    train_eval_path = params.train_eval_path
    test_path = params.test_path
    test_eval_path = params.test_eval_path
    
    estimate_path = f"{test_path}"
    estimate_eval_path = f"{test_eval_path}"
    
    if params.learning ==1: 
        print("train_data_preprocessing_start")
        #trainデータの前処理
        train_x,train_y,file_names,_,_,vx,vy,inlet_left_list,inlet_right_list,outlet_left_list,outlet_right_list = Preprocessing_standard_mix_for(train_path,train_eval_path,params.width,params.standard,params.cut,mode="train")
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        vx = torch.tensor(vx, dtype=torch.float32)
        vy = torch.tensor(vy, dtype=torch.float32)
        inlet_left_list = torch.tensor(inlet_left_list, dtype=torch.float32)
        inlet_right_list = torch.tensor(inlet_right_list, dtype=torch.float32)
        outlet_left_list = torch.tensor(outlet_left_list, dtype=torch.float32)
        outlet_right_list = torch.tensor(outlet_right_list, dtype=torch.float32)
        train_x, train_y,vx,vy = train_x[:params.cut_size], train_y[:params.cut_size],vx[:params.cut_size],vy[:params.cut_size]
        inlet_left_list,inlet_right_list = inlet_left_list[:params.cut_size],inlet_right_list[:params.cut_size]
        trainset = torch.utils.data.TensorDataset(train_x,train_y,vx,vy,inlet_left_list,inlet_right_list,outlet_left_list,outlet_right_list)
        dataloader = torch.utils.data.DataLoader(trainset,batch_size = params.batch_size, num_workers = 2, drop_last=True)

        #testデータの前処理
        test_x,test_y,file_names_test,avg_list,std_list,vx_test,vy_test,inlet_left_list_test,inlet_right_list_test,outlet_left_list_test,outlet_right_list_test = Preprocessing_standard_mix_for(estimate_path,estimate_eval_path,params.width,params.standard,params.cut,mode="eval")
        test_x = torch.tensor(test_x, dtype=torch.float32)
        test_y = torch.tensor(test_y, dtype=torch.float32)
        vx_test = torch.tensor(vx_test, dtype=torch.float32)
        vy_test = torch.tensor(vy_test, dtype=torch.float32)
        inlet_left_list_test = torch.tensor(inlet_left_list_test, dtype=torch.float32)
        inlet_right_list_test = torch.tensor(inlet_right_list_test, dtype=torch.float32)
        outlet_left_list_test = torch.tensor(outlet_left_list_test, dtype=torch.float32)
        outlet_right_list_test = torch.tensor(outlet_right_list_test, dtype=torch.float32)
        testset = torch.utils.data.TensorDataset(test_x,test_y,vx_test,vy_test,inlet_left_list_test,inlet_right_list_test,outlet_left_list_test,outlet_right_list_test)
        test_loader= torch.utils.data.DataLoader(testset,batch_size = params.test_batch_size, num_workers = 2, drop_last=True)

    print(f"{estimate_path=}")
    print(f"{estimate_eval_path=}")
    eval_x,eval_y,file_names_estimate,avg_list,std_list,vx_eval,vy_eval,inlet_left_list_eval,inlet_right_list_eval,outlet_left_list_eval,outlet_right_list_eval = Preprocessing_standard_mix_for(estimate_path,estimate_eval_path,params.width,params.standard,params.cut,mode="eval")
    eval_x = torch.tensor(eval_x, dtype=torch.float32)
    eval_y = torch.tensor(eval_y, dtype=torch.float32)
    vx_eval = torch.tensor(vx_eval, dtype=torch.float32)
    vy_eval = torch.tensor(vy_eval, dtype=torch.float32)
    inlet_left_list_eval = torch.tensor(inlet_left_list_eval, dtype=torch.float32)
    inlet_right_list_eval = torch.tensor(inlet_right_list_eval, dtype=torch.float32)
    outlet_left_list_eval = torch.tensor(outlet_left_list_eval, dtype=torch.float32)
    outlet_right_list_eval = torch.tensor(outlet_right_list_eval, dtype=torch.float32)
    evalset = torch.utils.data.TensorDataset(eval_x,eval_y,vx_eval,vy_eval,inlet_left_list_eval,inlet_right_list_eval,outlet_left_list_eval,outlet_right_list_eval)
    estimate_loader= torch.utils.data.DataLoader(evalset,batch_size = 1, num_workers = 2, drop_last=True)
    if params.learning == 1:
        file_maker(f"{params.root_path}/result/{params.output_path}")
        condition_text(params.message,params.output_path,params)
        UNet_ = UNet2()

        #modelの定義
        #model = modules.Input_Mix_divide_skip_detailed_12(UNet=UNet_).to(device)
        model = modules.Input_Mix_divide_skip_detailed_18(UNet=UNet_).to(device)
        #model = modules.To_high_resolution(UNet=UNet_high_resolution).to(device)
        #loss_weights = mymodule.LossWeights().to(device)
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
        #optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_weights.parameters()),lr=params.lr)
        #loss_fn = torch.nn.MSELoss()
        loss_fn = mymodule.compute_losses
        if params.additional_learning: 
            model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, f"{params.root_path}/result/{params.output_path}/save_temp_weight/checkpoint_{params.output_path}_epoch={params.additional_epoch-1}.pth")
        else:
            file_maker(f"{params.root_path}/result/{params.output_path}")
            condition_text(params.message,params.output_path,params)
            start_epoch = 0
        loss_list = []
        loss_list_test = []

        # training
        epoch_bar = tqdm(range(start_epoch, params.epochs+1))
        for epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch:{epoch}")
            iter_bar = tqdm(dataloader, leave=False)
            train_loss = 0
            test_loss = 0
            avg_test_loss = 0
            model.train()
            for iter, (x,y,vx,vy,inlet_l,inlet_r,outlet_l,outlet_r) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                vx,vy = vx.to(device),vy.to(device)
                vx = vx.unsqueeze(1)
                vy = vy.unsqueeze(1)
                inlet_l,inlet_r = inlet_l.unsqueeze(1),inlet_r.unsqueeze(1)
                outlet_l,outlet_r = outlet_l.unsqueeze(1),outlet_r.unsqueeze(1)
                inlet_l,inlet_r = inlet_l.to(device),inlet_r.to(device)
                outlet_l,outlet_r = outlet_l.to(device),outlet_r.to(device)
                v = torch.cat((vx,vy,inlet_l,inlet_r,outlet_l,outlet_r),dim=1)
                v_loss = torch.cat((vx,vy),dim=1).detach()
                #損失関数に速度を渡すために計算グラフから速度vを切り離したものを作成
                out = model(v)
                loss,losses = loss_fn(out,y,v_loss)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_bar.set_postfix({"loss=": f"{loss.item():.2e}"})

            # テストデータでの損失を計算
            model.eval()
            with torch.no_grad():  # テスト時は勾配計算を行わない
                for iter, (x,y,vx,vy,inlet_l,inlet_r,outlet_l,outlet_r) in enumerate(test_loader):
                    x = x.to(device)
                    y = y.to(device)
                    vx,vy = vx.to(device),vy.to(device)
                    vx = vx.unsqueeze(1)
                    vy = vy.unsqueeze(1)
                    inlet_l,inlet_r = inlet_l.unsqueeze(1),inlet_r.unsqueeze(1)
                    outlet_l,outlet_r = outlet_l.unsqueeze(1),outlet_r.unsqueeze(1)
                    inlet_l,inlet_r = inlet_l.to(device),inlet_r.to(device)
                    outlet_l,outlet_r = outlet_l.to(device),outlet_r.to(device)
                    v = torch.cat((vx,vy,inlet_l,inlet_r,outlet_l,outlet_r),dim=1)
                    v_loss = torch.cat((vx,vy),dim=1).detach()
                    out = model(v)
                    loss,losses = loss_fn(out,y,v_loss)
                    test_loss += loss.item()
                    iter_bar.set_postfix({"loss=": f"{loss.item():.2e}"})

            # 損失の平均を計算
            avg_train_loss = train_loss/(len(train_x)//params.batch_size)
            avg_test_loss = test_loss/(len(test_x)//params.test_batch_size)
            loss_list.append(avg_train_loss)
            loss_list_test.append(avg_test_loss)
            epoch_bar.set_postfix({"train_loss": f"{avg_train_loss:.2e}", "test_loss": f"{avg_test_loss:.2e}"})
            if epoch % params.save_interval == 0 and epoch != 0:    
                file_maker(f"{params.root_path}/result/{params.output_path}/save_temp_weight")
                file_maker(f"{params.root_path}/result/{params.output_path}/loss_file")
                save_checkpoint(model, optimizer, epoch, loss, f"{params.root_path}/result/{params.output_path}/save_temp_weight/checkpoint_{params.output_path}_epoch={epoch}.pth")
                torch.save(model,f"{params.root_path}/result/{params.output_path}/weight_{params.output_path}_epoch={epoch}.pth")
        torch.save(model,f"{params.root_path}/result/{params.output_path}/weight_{params.output_path}.pth")
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
        plt.savefig(f"{params.root_path}/result/{params.output_path}/loss_{params.output_path}.pdf") 
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
            plt.savefig(f"{params.root_path}/result/{params.output_path}/loss_{params.output_path}_from_100.pdf") 

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
    count = 0
    output_list = []
    model.eval()
    avg_list = np.reshape(avg_list,[-1,1])
    std_list = np.reshape(std_list,[-1,1])
    for counter,(data,evaly,vx_eval,vy_eval,inlet_l_eval,inlet_r_eval,outlet_l_eval,outlet_r_eval) in enumerate(estimate_loader):
            vx_eval,vy_eval = vx_eval.to(device),vy_eval.to(device)
            inlet_l_eval,inlet_r_eval = inlet_l_eval.to(device),inlet_r_eval.to(device)
            outlet_l_eval,outlet_r_eval = outlet_l_eval.to(device),outlet_r_eval.to(device)
            vx_eval = vx_eval.unsqueeze(1)
            vy_eval = vy_eval.unsqueeze(1)
            inlet_l_eval,inlet_r_eval = inlet_l_eval.unsqueeze(1),inlet_r_eval.unsqueeze(1)
            outlet_l_eval,outlet_r_eval = outlet_l_eval.unsqueeze(1),outlet_r_eval.unsqueeze(1)
            v = torch.cat((vx_eval,vy_eval,inlet_l_eval,inlet_r_eval,outlet_l_eval,outlet_r_eval),dim=1)
            p = model(v)
            p_output = p.detach().cpu().numpy()
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
        file_maker(f"{params.root_path}/result/{params.output_path}")
        file_maker(f"{params.root_path}/result/{params.output_path}/{params.file_path}")
        if params.byepoch:
            file_maker(f"{params.root_path}/result/{params.output_path}/{params.file_path_byepoch}")
            pd.DataFrame(out,columns=["X Velocity","Y Velocity"]).to_csv(f"{params.root_path}/result/{params.output_path}/{params.file_path_byepoch}/estimate_{file_names_estimate[i]}.csv", index=False)
        else:
            pd.DataFrame(out,columns=["X Velocity","Y Velocity"]).to_csv(f"{params.root_path}/result/{params.output_path}/{params.file_path}/estimate_{file_names_estimate[i]}.csv", index=False)

@dataclass
class HyperParameters:
    #ファイル関連
    root_path = "/mnt/data1/tony"
    task_name: str = "estimate_velocity"
    output_path: str = "pinns_estimate_1109"
    message: str = "PINNsによる速度推定。1000からは重みを変えて追加学習"
    file_path: str = "inlet_value1" #推定に使うデータのフォルダ
    train_file_path = "inlet_value1" #学習データのフォルダ
    train_path: str = f"{root_path}/{train_file_path}" #学習データ
    train_eval_path: str  = f"{root_path}/{train_file_path}" #学習データの正解ラベル
    test_path: str = f"{root_path}/{file_path}" #推定に使うデータ
    test_eval_path: str  = f"{root_path}/{file_path}" #推定に使うデータ(意味ない)
    weight_eval_path = f"{root_path}/result/{output_path}/weight_{output_path}.pth" #学習済みモデルの名前
    
    #ハイパーパラメーター
    cut_size: int = 300000 #訓練データのサイズ(実際には10%はテストデータとして使う。全て使う時は大きい数を指定)
    save_interval: int = 50 #何エポックごとにモデルを保存するか
    learning = 0 #1で学習を行う,0で学習を行わずに推定のみを行う
    standard = 0 #1で標準化を行う,0で行わない
    epochs: int = 2000 #エポック数
    width: int = 32
 #画像の幅
    batch_size: int = 256 #バッチサイズ
    test_batch_size: int = 4
    lr: float = 1.0e-3 #学習率
    time_steps: int =  1000  # T もう少し小さくても良いはず,何回ノイズを加えるか
    end_estimate_number=10000 #推定するデータの数(多すぎる推定データを与えたときにこの数で推定をやめる)
    rate = 0.1 #訓練データとテストデータの割合(前処理が終わっているデータの何割をテストデータとして使うか)
    cut = 0.5 #cut以下の速度の値を0にする(学習を簡単にするために一定以下の速度を切り落とす,切り落とさない時は0を指定,0,5ぐらいで対象以外の部分を除ける)

    byepoch = True #学習途中のファイルで推定するならTrue
    #target_epoch: int = 350 #どのエポックのモデルを使って推定するか
    target_epoch: int = 1000 #どのエポックのモデルを使って推定するか
    weight_eval_path_byepoch = f"../../result/{output_path}/weight_{output_path}_epoch={target_epoch}.pth" #学習済みモデルの名前
    file_path_byepoch: str = f"{file_path}_epoch_{target_epoch}" #推定に使うデータのフォルダ

    additional_learning = False #Trueですでに保存されている重みを読み込んで学習を再開する
    additional_epoch = 1000 #学習を再開するエポック数(重みファイルのepoch+1)
     
params = HyperParameters()
ddpm_train(params)
