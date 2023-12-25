import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import re
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
import mymodule #import Net1D
from mymodule import atoi
from mymodule import natural_keys
from mymodule import Preprocessing
import time
import pandas as pd
from mymodule import file_maker
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

#hyper_parameter
batch = 128
types ="sim" #beads or sim or rhodamine
learning = 1 #1=learning 
epochs = 500
dmax = 330
dmin = 230
model_name = "CNN"
model_path = mymodule.Complicated2DCNN()

#setup
weight_save_by_epoch = "off"

#path 
weight_path = "CNN_2D_complicated"

train_path = "../data_train/Time=1"
train_eval_path = "../data_train/Time=20"
test_path = "../Time=1"
test_eval_path = "../Time=20"

estimate_path = f"{train_path}"
estimate_eval_path = f"{train_eval_path}"

if learning == 0:
    save_path ="train"

output_path = f"{weight_path}"
output_evaluate_path = f"{weight_path}"
#device = "cpu"
device = "cuda:0"
file_maker(f"../weight_by_epoch/{weight_path}")
file_maker(f"../result/{weight_path}")

if learning == 1:
    print("train_data_preprocessing_start")
    train_x,train_y,_,_,_ = Preprocessing(train_path,train_eval_path,dmax,dmin)
    nan_indices = np.where(np.isnan(train_x))[0]
    
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    trainset = torch.utils.data.TensorDataset(train_x,train_y)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size = batch, shuffle = True, num_workers = 2, drop_last=True)
    
    test_x,test_y,_,_,_ = Preprocessing(test_path,test_eval_path,dmax,dmin)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)
    testset = torch.utils.data.TensorDataset(test_x,test_y)
    testloader = torch.utils.data.DataLoader(testset,batch_size = batch, shuffle = True, num_workers = 2, drop_last=True)

print("evaluate_data_preprocessing_start")
evaluate_x,evaluate_y,file_names,avg,std = Preprocessing(estimate_path,estimate_eval_path,dmax,dmin)
evalate_y_copy = np.copy(evaluate_y)
evaluate_y_copy = torch.tensor(evaluate_y, dtype=torch.float32)
evaluate_x = torch.tensor(evaluate_x, dtype=torch.float32)
evaluate_y = torch.tensor(evaluate_y, dtype=torch.float32)
evalset = torch.utils.data.TensorDataset(evaluate_x,evaluate_y)
evalloader = torch.utils.data.DataLoader(evalset,batch_size = batch,num_workers = 2, drop_last=True)


if learning == 1:
    #Learning
    model = model_path
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loss_list=[]
    loss_list_test = []
    print("learning_start")
    for epoch in range(epochs):
        start =time.time()
        #torch.cuda.synchronize()
        train_loss = 0
        test_loss = 0
        avg_test_loss = 0
        model.train()
        loss_count=0
        for train_counter,(inputs,labels) in enumerate(trainloader):
            inputs,labels = inputs.to(device),labels.to(device)
            out = model(inputs)
            loss = criterion(out,labels)
            #print(f"out={out[:5]}")
            #print(f"labels={labels[:5]}")
            """if torch.isnan(loss).any():
                print(f"Nan_loss={loss}")
                print(f"train_counter={train_counter}")
            print(f"loss={loss}")
            if train_counter == 5:
                sys.exit()"""
            train_loss += loss.item()
            #train_loss += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        for test_counter,(inputs,labels) in enumerate(testloader):
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out,labels)
            test_loss += loss.item()
        print(f"train_loss={train_loss}")
        avg_train_loss = train_loss/len(train_x)
        avg_test_loss = test_loss/len(test_x)
        loss_list.append(avg_train_loss)
        loss_list_test.append(avg_test_loss)
        print("epochs="+str(epoch+1)+",avg_train_loss="+str(avg_train_loss)+",avg_test_loss="+str(avg_test_loss))
        #torch.cuda.synchronize()
        if weight_save_by_epoch == "on" and epoch%5==0:
            torch.save(model,"weight_by_epoch/"+str(weight_path)+"/weight_"+weight_path+"_epoch"+str(1+epoch)+".pth")
        elapsed_time = time.time() -start
        print(elapsed_time,"sec.")
    
    fig=plt.figure()
    plt.plot(loss_list,label='valid', lw=2, c='b')
    plt.plot(loss_list_test,label='test', lw=2, c='k')
    plt.grid()
    plt.rcParams["font.size"] = 10
    plt.xticks(np.arange(0, epochs,5))
    plt.xlabel("Epoch")
    plt.ylabel("Loss function")
    plt.legend()
    plt.savefig(f"../result/{weight_path}/loss_{output_path}.pdf")
    torch.save(model,f"../result/{weight_path}/weight_{weight_path}.pth")

#evaluate_step
print("evaluate_step_start")
weight_eval_path = f"../result/{weight_path}/weight_{weight_path}.pth"
model = torch.load(weight_eval_path)
model = model.to(device)
model.eval()
label_list = []
pred_list = []
pred_list_2  = []

print("estimate_start")
eval_loss = 0
loss_list = []
#criterion = nn.MSELoss()
criterion = nn.MSELoss()
count = 0
output_list = []
for data,evaly in evalloader:
        data = data.to(device)
        evaly = evaly.to(device)
        p = model(data)
        p_output = p.detach().cpu().numpy()
        #p_output *= std[count:count+batch].reshape(-1,1,1,1)
        #p_output += avg[count:count+batch].reshape(-1,1,1,1)
        output_list.append(p_output)
        #count+=batch
output_list = np.array(output_list)
list_shape=len(evaluate_x)-(len(evaluate_x)%batch)
output_list = np.reshape(output_list,[list_shape,2,-1])
print(output_list.shape)
file_maker(f"../result/{weight_path}/estimate_result")
for i in tqdm(range(len(output_list)),total=len(output_list)):
    out = output_list[i]
    out = out.T
    if learning == 0:
        file_maker(f"../result/{weight_path}/{save_path}")
        pd.DataFrame(out,columns=["X Velocity","Y Velocity"]).to_csv(f"../result/{weight_path}/{save_path}/estimate_{file_names[i]}.csv", index=False)
    else:
        pd.DataFrame(out,columns=["X Velocity","Y Velocity"]).to_csv(f"../result/{weight_path}/estimate_result/estimate_{file_names[i]}.csv", index=False)

