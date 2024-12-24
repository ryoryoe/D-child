import numpy as np
import csv
import re
import math
import os
import pickle
import glob
import copy
import scipy
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sys
from pympler import muppy, summary,asizeof

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
def atoi(text):
        return int(text) if text.isdigit() else text
def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]
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

def process_file_mix(path, eval_path,cut=0):
    df = pd.read_csv(path)
    df = df.values
    df = df.astype(float)
    #df配列の中で0.5以下の要素を0に変換
    #df = np.where(df <= cut, 0, df)
    df_eval = pd.read_csv(eval_path)
    df_eval = df_eval.values
    df_eval = df_eval.astype(float)
    #df_eval = np.where(df_eval <= cut, 0, df_eval)
    return df,df_eval

def process_file(path, eval_path,cut=0):
    df = pd.read_csv(path)
    df = df.values
    df = df.astype(float)
    #df配列の中で0.5以下の要素を0に変換
    #df = np.where(df <= cut, 0, df)
    df_eval = pd.read_csv(eval_path)
    df_eval = df_eval.values
    df_eval = df_eval.astype(float)
    #df_eval = np.where(df_eval <= cut, 0, df_eval)
    return df,df_eval

def standardization(velocity_x, velocity_y):
    avg = np.mean(velocity_x)
    std = np.std(velocity_x)
    avg_eval = np.mean(velocity_y)
    std_eval = np.std(velocity_y)
    velocity_x -= avg
    velocity_x /= std
    velocity_y -=avg_eval
    velocity_y /= std_eval
    return velocity_x,velocity_y,avg,std

def Preprocessing_standard_3D(inputname,input_evalname,width,standard=0,cut=0,v2=False,z_width=4): #入り口の数が2つなら2v=True
    input_path = sorted([entry.path for entry in os.scandir(inputname) if entry.is_file() and entry.name.endswith('.csv')],
    key=natural_keys)
    eval_path = sorted(glob.glob(input_evalname + "/*.csv"), key=natural_keys)
    num_cores = os.cpu_count()
    print("並列処理を開始します")
    batch_size = 5000
    spatial = 4 #データを何個間隔で間引くか(4回の追加学習を行う)
    repeat_number = 3 #今が何回目か(0が最初)

    results = []
    train_list = []
    eval_list = []
    file_names_list = []
    avg_lists = []
    std_lists = []
    vx_lists = []
    vy_lists = []
    vx2_lists = []
    vy2_lists = []
    vz_lists = []
    vz2_lists = []
    
    for i in range(0, len(input_path),batch_size):
        #batch_files = input_path[i+repeat_number:i+batch_size]
        #batch_eval_files = eval_path[i+repeat_number:i+batch_size]
        batch_files = input_path[i+repeat_number:i+batch_size:spatial]
        batch_eval_files = eval_path[i+repeat_number:i+batch_size:spatial]
        results = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            batch_results = list(tqdm(executor.map(process_file, batch_files, batch_eval_files, [cut]*len(batch_files)), total=len(batch_files)))
            results.extend(batch_results)

        #with ProcessPoolExecutor(max_workers=num_cores) as executor:
        #    results = list(tqdm(executor.map(process_file, input_path,eval_path,[cut]*len(input_path)),total=len(input_path)))
        train,evals = zip(*results)
        train = np.asarray(train, dtype=float)
        evals = np.asarray(evals, dtype=float)
        train = np.reshape(train, [len(train),width*width*z_width,3])
        evals = np.reshape(evals, [len(evals),width*width*z_width,3])
        print(f"除去する前のデータ数_{len(train)=}")
        #delete nan file
        nan_indices = np.where(np.any(np.isnan(evals),axis=(1,2)))[0]
        print(f"{len(nan_indices)=}")
        greater_than_two = np.any(evals >= 20, axis=(1, 2))
        print(f"{len(greater_than_two)=}")
        indices_to_remove = np.where(greater_than_two)[0]
        indices_to_remove = np.concatenate([indices_to_remove,nan_indices])
        indices_to_remove = np.sort(indices_to_remove)
        #print(f"before_{len(train)=}")
        train = np.delete(train,indices_to_remove,axis=0)
        evals = np.delete(evals,indices_to_remove,axis=0)

        #get_file_name and delete nan file
        file_names = file_name_maker(f"{inputname}")
        file_names = sort_and_combine_strings(file_names)
        file_names = file_names[i+repeat_number:i+batch_size:spatial]
        file_names = np.delete(file_names, indices_to_remove)
        # 正規表現を使用してvxとvyを抽出
        vx_list = []
        vy_list = []
        vz_list = []
        vx2_list = []
        vy2_list = []
        vz2_list = []
        for filename in file_names:
            vx_match = re.search(r"velocity_x=([-0-9.]+)", filename)
            vy_match = re.search(r"velocity_y=([-0-9.]+)", filename)
            vz_match = re.search(r"velocity_z=([-0-9.]+)", filename)
            if vx_match and vy_match:
                vx = float(vx_match.group(1))
                vy = float(vy_match.group(1))
                try:
                    vz = float(vz_match.group(1))            
                except:
                    vz = 0.000
            else:
                print("速度を抽出できませんでした") 
                sys.exit()
            vx_list.append(vx)
            vy_list.append(vy)
            vz_list.append(vz)

            if v2:
                vx_match = re.search(r"velocity_x2=([-0-9.]+)", filename)
                vy_match = re.search(r"velocity_y2=([-0-9.]+)", filename)
                vz_match = re.search(r"velocity_z2=([-0-9.]+)", filename)
                if vx_match and vy_match:
                    vx2 = float(vx_match.group(1))
                    vy2 = float(vy_match.group(1))
                    vz2 = float(vz_match.group(1))
                else:
                    print("速度を抽出できませんでした") 
                    sys.exit()
                vx2_list.append(vx2)
                vy2_list.append(vy2)
                vz2_list.append(vz2)
        # 0.4以上の値を持つインデックスを見つけて削除
        #indices_to_remove = [i for i, value in enumerate(vy_list) if value < 0.4]
        indices_to_remove = [i for i, value in enumerate(vx_list) if value < -0.3 or value > 0.3]
        #indices_to_remove = [i for i, value in enumerate(vx_list) if value > -0.3 and value < 0.3]
        #indices_to_remove = [i for i, value in enumerate(vx_list) if value>=0]
        # indices_to_removeを逆順にして、vyとvxから要素を削除
        # 逆順にしないと、削除する際にインデックスがずれる可能性がある
        for index in sorted(indices_to_remove, reverse=True):
            del vy_list[index]
            del vx_list[index]
            del vz_list[index]
        file_names = np.delete(file_names, indices_to_remove)    
        print(f"{train.shape=}")
        train = np.delete(train, indices_to_remove, axis=0)
        evals = np.delete(evals, indices_to_remove, axis=0) 

        file_names = file_names.tolist()
        #standardization
        if standard == 1:
            train = np.reshape(train, [len(train),-1])
            evals = np.reshape(evals, [len(evals),-1])
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                results = list(tqdm(executor.map(standardization, train,evals),total=len(train)))
            train,evals,avg_list,std_list = zip(*results)
            train = np.reshape(train, [len(train),width*width*z_width,3])
            evals = np.reshape(evals, [len(evals),width*width*z_width,3])
        else:
            avg_list = 0
            std_list = 0 
        # data adjustment
        print(f"除去した後のデータ数_{len(train)=}")
        train = np.reshape(train, [-1,width,width,z_width,3]).transpose(0,4,1,2,3)
        evals = np.reshape(evals, [-1,width,width,z_width,3]).transpose(0,4,1,2,3)
        train = train[:len(evals)]
        file_names = file_names[:len(evals)]
        train_list.extend(train)
        eval_list.extend(evals)
        file_names_list.extend(file_names)
        vx_lists.extend(vx_list)
        vy_lists.extend(vy_list)
        vz_lists.extend(vz_list)
        print(f"途中の{len(train_list)=}")
        #if (i+batch_size) >= 180000:
        #    break
    train_list = np.asarray(train_list, dtype=float)
    eval_list = np.asarray(eval_list, dtype=float)
    vx_lists = np.asarray(vx_lists, dtype=float)
    vy_lists = np.asarray(vy_lists, dtype=float)
    vz_lists = np.asarray(vz_lists, dtype=float)
    #前のspatial処理(今は冒頭で処理)
    #train_list = train_list[::spatial]
    #eval_list = eval_list[::spatial]
    #file_names_list = file_names_list[::spatial]
    #vx_lists = vx_lists[::spatial]
    #vy_lists = vy_lists[::spatial]
    #vz_lists = vz_lists[::spatial]
    print("並列処理を終了します")
    print(f"len(train_list)={len(train_list)}")
    print(f"{file_names_list[:5]=}")
    print(f"{vx_lists[:5]=}")
    print(f"{vy_lists[:5]=}")
    if v2:
        return train, evals,file_names,avg_list,std_list,vx_list,vy_list,vx2_list,vy2_list,vz_list,vz2_list
    return train_list, eval_list,file_names_list,avg_list,std_list,vx_lists,vy_lists,vz_lists

def Preprocessing_standard(inputname,input_evalname,width,standard=0,cut=0,v2=False): #入り口の数が2つなら2v=True
    input_path = sorted(glob.glob(inputname + "/*.csv"), key=natural_keys)
    eval_path = sorted(glob.glob(input_evalname + "/*.csv"), key=natural_keys)
    num_cores = os.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_file, input_path,eval_path,[cut]*len(input_path)),total=len(input_path)))
    train,evals = zip(*results)
    train = np.asarray(train, dtype=float)
    evals = np.asarray(evals, dtype=float)
    train = np.reshape(train, [len(train),width*width,2])
    evals = np.reshape(evals, [len(evals),width*width,2])
    
    #delete nan file
    nan_indices = np.where(np.any(np.isnan(evals),axis=(1,2)))[0]
    greater_than_two = np.any(evals >= 2, axis=(1, 2))
    indices_to_remove = np.where(greater_than_two)[0]
    indices_to_remove = np.concatenate([indices_to_remove,nan_indices])
    indices_to_remove = np.sort(indices_to_remove)
    #print(f"before_{len(train)=}")
    train = np.delete(train,indices_to_remove,axis=0)
    evals = np.delete(evals,indices_to_remove,axis=0)

    #get_file_name and delete nan file
    file_names = file_name_maker(f"{inputname}")
    file_names = sort_and_combine_strings(file_names)
    file_names = np.delete(file_names, indices_to_remove)
    # 正規表現を使用してvxとvyを抽出
    vx_list = []
    vy_list = []
    vx2_list = []
    vy2_list = []
    for filename in file_names:
        vx_match = re.search(r"velocity_x=([-0-9.]+)", filename)
        vy_match = re.search(r"velocity_y=([-0-9.]+)", filename)
        
        if vx_match and vy_match:
            vx = float(vx_match.group(1))
            vy = float(vy_match.group(1))
        else:
            print("速度を抽出できませんでした") 
            sys.exit()
        vx_list.append(vx)
        vy_list.append(vy)

        if v2:
            vx_match = re.search(r"velocity_x2=([-0-9.]+)", filename)
            vy_match = re.search(r"velocity_y2=([-0-9.]+)", filename)
            if vx_match and vy_match:
                vx2 = float(vx_match.group(1))
                vy2 = float(vy_match.group(1))
            else:
                print("速度を抽出できませんでした") 
                sys.exit()
            vx2_list.append(vx2)
            vy2_list.append(vy2)
    # 0.4以上の値を持つインデックスを見つけて削除
    indices_to_remove = [i for i, value in enumerate(vy_list) if value > 0.8]
    # indices_to_removeを逆順にして、vyとvxから要素を削除
    # 逆順にしないと、削除する際にインデックスがずれる可能性がある
    for index in sorted(indices_to_remove, reverse=True):
        del vy_list[index]
        del vx_list[index]
        if v2:
            del vx2_list[index]
            del vy2_list[index]
    file_names = np.delete(file_names, indices_to_remove)    
    train = np.delete(train, indices_to_remove, axis=0)
    evals = np.delete(evals, indices_to_remove, axis=0) 

    file_names = file_names.tolist()
    #standardization
    if standard == 1:
        train = np.reshape(train, [len(train),-1])
        evals = np.reshape(evals, [len(evals),-1])
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(tqdm(executor.map(standardization, train,evals),total=len(train)))
        train,evals,avg_list,std_list = zip(*results)
        train = np.reshape(train, [len(train),width*width,2])
        evals = np.reshape(evals, [len(evals),width*width,2])
    else:
        avg_list = 0
        std_list = 0 
    # data adjustment
    train = np.reshape(train, [-1,width,width,2]).transpose(0,3,1, 2)
    evals = np.reshape(evals, [-1,width,width,2]).transpose(0,3,1,2)
    train = train[:len(evals)]
    file_names = file_names[:len(evals)]
    if v2:
        return train, evals,file_names,avg_list,std_list,vx_list,vy_list,vx2_list,vy2_list
    return train, evals,file_names,avg_list,std_list,vx_list,vy_list

def sort_file(inputname,input_evalname,width,standard=0,cut=0):
    input_path = sorted(glob.glob(inputname + "/*.csv"), key=natural_keys)
    eval_path = sorted(glob.glob(input_evalname + "/*.csv"), key=natural_keys)
    num_cores = os.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_file_mix, input_path,eval_path,[cut]*len(input_path)),total=len(input_path)))
    train,evals = zip(*results)
    train = np.asarray(train, dtype=float)
    evals = np.asarray(evals, dtype=float)
    train = np.reshape(train, [len(train),width*width,2])
    evals = np.reshape(evals, [len(evals),width*width,2])
    
    #delete nan file
    nan_indices = np.where(np.any(np.isnan(evals),axis=(1,2)))[0]
    greater_than_two = np.any(evals >= 2, axis=(1, 2))
    indices_to_remove = np.where(greater_than_two)[0]
    indices_to_remove = np.concatenate([indices_to_remove,nan_indices])
    indices_to_remove = np.sort(indices_to_remove)
    #print(f"before_{len(train)=}")
    train = np.delete(train,indices_to_remove,axis=0)
    evals = np.delete(evals,indices_to_remove,axis=0)

    #get_file_name and delete nan file
    file_names = file_name_maker(f"{inputname}")
    file_names = sort_and_combine_strings(file_names)
    file_names = np.delete(file_names, indices_to_remove)
    # 正規表現を使用してvxとvyを抽出
    vx_list = []
    vy_list = []
    vx2_list = []
    vy2_list = []
    for filename in file_names:
        vx_match = re.search(r"velocity_x=([-0-9.]+)", filename)
        vy_match = re.search(r"velocity_y=([-0-9.]+)", filename)
        
        if vx_match and vy_match:
            vx = float(vx_match.group(1))
            vy = float(vy_match.group(1))
        else:
            print("速度を抽出できませんでした") 
            sys.exit()
        vx_list.append(vx)
        vy_list.append(vy)

        vx_match = re.search(r"velocity_x2=([-0-9.]+)", filename)
        vy_match = re.search(r"velocity_y2=([-0-9.]+)", filename)
        if vx_match and vy_match:
            vx2 = float(vx_match.group(1))
            vy2 = float(vy_match.group(1))
        else:
            vx2 = 0.000
            vy2 = 0.000
        vx2_list.append(vx2)
        vy2_list.append(vy2)
    # 0.4以上の値を持つインデックスを見つけて削除
    indices_to_remove = [i for i, value in enumerate(vy_list) if value < 0.4]
    # indices_to_removeを逆順にして、vyとvxから要素を削除
    # 逆順にしないと、削除する際にインデックスがずれる可能性がある
    for index in sorted(indices_to_remove, reverse=True):
        del vy_list[index]
        del vx_list[index]
        del vx2_list[index]
        del vy2_list[index]
    file_names = np.delete(file_names, indices_to_remove)    
    train = np.delete(train, indices_to_remove, axis=0)
    evals = np.delete(evals, indices_to_remove, axis=0) 

    file_names = file_names.tolist()
    #standardization
    if standard == 1:
        train = np.reshape(train, [len(train),-1])
        evals = np.reshape(evals, [len(evals),-1])
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(tqdm(executor.map(standardization, train,evals),total=len(train)))
        train,evals,avg_list,std_list = zip(*results)
        train = np.reshape(train, [len(train),width*width,2])
        evals = np.reshape(evals, [len(evals),width*width,2])
    else:
        avg_list = 0
        std_list = 0 
    # data adjustment
    train = np.reshape(train, [-1,width,width,2]).transpose(0,3,1, 2)
    evals = np.reshape(evals, [-1,width,width,2]).transpose(0,3,1,2)
    train = train[:len(evals)]
    file_names = file_names[:len(evals)]
    return train, evals,file_names,avg_list,std_list,vx_list,vy_list,vx2_list,vy2_list

def sort_file_mix_for(inputname,input_evalname,width,standard=0,cut=0,inlet_min=9,inlet_max=23):
    input_path = sorted(glob.glob(inputname + "/*.csv"), key=natural_keys)
    eval_path = sorted(glob.glob(input_evalname + "/*.csv"), key=natural_keys)
    num_cores = os.cpu_count()
    batch_size = 5000
    train_list,eval_list,file_names_list,vx_lists,vy_lists,inlet_left_lists,inlet_right_lists,outlet_left_lists,outlet_right_lists = [],[],[],[],[],[],[],[],[]
    for i in range(0, len(input_path),batch_size):
        batch_files = input_path[i:i+batch_size]
        batch_eval_files = eval_path[i:i+batch_size]
        results = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            #results = list(tqdm(executor.map(process_file_mix, input_path,eval_path,[cut]*len(input_path)),total=len(input_path)))
            results = list(tqdm(executor.map(process_file_mix, batch_files, batch_eval_files, [cut]*len(batch_files)), total=len(batch_files)))
        train,evals = zip(*results)
        #形状の違うデータのファイルネームも先に削除
        file_names = file_name_maker(f"{inputname}")
        file_names = sort_and_combine_strings(file_names)
        file_names = file_names[i:i+batch_size]
        train = list(train)
        evals = list(evals) 
        for i in range(len(train) - 1, -1, -1):  # 後ろから順に調べる
            if np.shape(train[i]) != (1024, 2) or np.shape(evals[i]) != (1024, 2):
            #if np.shape(train[i]) != (1024, 2): mix_learningの時
                print(f"del: {file_names[i]}")
                del train[i]
                del evals[i]
                del file_names[i]
        train = np.asarray(train, dtype=float)
        evals = np.asarray(evals, dtype=float)
        #train = np.reshape(train, [len(train),16*16,2]) #高解像度の時
        train = np.reshape(train, [len(train),width*width,2])
        evals = np.reshape(evals, [len(evals),width*width,2])
        
        #delete nan file
        nan_indices = np.where(np.any(np.isnan(evals),axis=(1,2)))[0]
        greater_than_two = np.any(evals >= 2, axis=(1, 2))
        indices_to_remove = np.where(greater_than_two)[0]
        indices_to_remove = np.concatenate([indices_to_remove,nan_indices])
        indices_to_remove = np.sort(indices_to_remove)
        #print(f"before_{len(train)=}")
        train = np.delete(train,indices_to_remove,axis=0)
        evals = np.delete(evals,indices_to_remove,axis=0)

        #get_file_name and delete nan file
        file_names = np.delete(file_names, indices_to_remove)
        # 正規表現を使用してvxとvyを抽出
        vx_list = []
        vy_list = []
        inlet_left_list = []
        inlet_right_list = []
        outlet_left_list = []
        outlet_right_list = []
        #velocity_x=-0.463_velocity_y=0.796_inlet_27_37_outlet_43_53
        for filename in file_names:
            vx_match = re.search(r"velocity_x=([-0-9.]+)", filename)
            vy_match = re.search(r"velocity_y=([-0-9.]+)", filename)

            inlet_match = re.search(r"inlet_(\d+)_(\d+)", filename)
            outlet_match = re.search(r"outlet_(\d+)_(\d+)", filename)

            if vx_match and vy_match:
                vx = float(vx_match.group(1))
                vy = float(vy_match.group(1))
            else:
                print("速度を抽出できませんでした") 
                sys.exit()
            vx_list.append(vx)
            vy_list.append(vy)
            
            if inlet_match and outlet_match:
            #if inlet_left_match and outlet_left_match:
                inlet_left = int(inlet_match.group(1))
                inlet_right = int(inlet_match.group(2))
                #inletをinlet_min,inlet_maxで正規化
                inlet_left = (inlet_left - inlet_min) / (inlet_max - inlet_min)
                inlet_right = (inlet_right - inlet_min) / (inlet_max - inlet_min)
                outlet_left = int(outlet_match.group(1))
                outlet_right = int(outlet_match.group(2))
            else:
                print("inletまたはoutletの値が見つかりませんでした")
                sys.exit()
            inlet_left_list.append(inlet_left)
            inlet_right_list.append(inlet_right)
            outlet_left_list.append(outlet_left)
            outlet_right_list.append(outlet_right)

        # 0.4以上の値を持つインデックスを見つけて削除
        indices_to_remove = [i for i, value in enumerate(vy_list) if value < 0.4]
        # indices_to_removeを逆順にして、vyとvxから要素を削除
        # 逆順にしないと、削除する際にインデックスがずれる可能性がある
        for index in sorted(indices_to_remove, reverse=True):
            del vy_list[index]
            del vx_list[index]
            del inlet_left_list[index]
            del inlet_right_list[index]
            del outlet_left_list[index]
            del outlet_right_list[index]
            
        file_names = np.delete(file_names, indices_to_remove)    
        train = np.delete(train, indices_to_remove, axis=0)
        evals = np.delete(evals, indices_to_remove, axis=0) 

        file_names = file_names.tolist()
        #standardization
        if standard == 1:
            train = np.reshape(train, [len(train),-1])
            evals = np.reshape(evals, [len(evals),-1])
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                results = list(tqdm(executor.map(standardization, train,evals),total=len(train)))
            train,evals,avg_list,std_list = zip(*results)
            train = np.reshape(train, [len(train),width*width,2])
            evals = np.reshape(evals, [len(evals),width*width,2])
        else:
            avg_list = 0
            std_list = 0 
        # data adjustment
        train = np.reshape(train, [-1,width,width,2]).transpose(0,3,1, 2)
        #train = np.reshape(train, [-1,16,16,2]).transpose(0,3,1, 2) #高解像度の時
        evals = np.reshape(evals, [-1,width,width,2]).transpose(0,3,1,2)
        train = train[:len(evals)]
        file_names = file_names[:len(evals)]
        train_list.extend(train)
        eval_list.extend(evals)
        file_names_list.extend(file_names)
        vx_lists.extend(vx_list)
        vy_lists.extend(vy_list)
        inlet_left_lists.extend(inlet_left_list)
        inlet_right_lists.extend(inlet_right_list)
        outlet_left_lists.extend(outlet_left_list)
        outlet_right_lists.extend(outlet_right_list)
        break
    #return train_list, eval_list,file_names,avg_list,std_list,vx_list,vy_list,inlet_left_list,inlet_right_list,outlet_left_list,outlet_right_list
    return train_list, eval_list,file_names_list,avg_list,std_list,vx_lists,vy_lists,inlet_left_lists,inlet_right_lists,outlet_left_lists,outlet_right_lists

def extract_velocities(path):
    match = re.search(r'velocity_x=([\d\.]+)_velocity_y=([\d\.]+)', path)
    if match:
        return match.group(1), match.group(2)
    return None, None

def sort_file_high_resolution(inputname,input_evalname,width,standard=0,cut=0,mode="train",inlet_min=9,inlet_max=23):
    input_path = sorted(glob.glob(inputname + "/*.csv"), key=natural_keys)
    eval_path = sorted(glob.glob(input_evalname + "/*.csv"), key=natural_keys)
    input_dict = {extract_velocities(path): path for path in input_path if extract_velocities(path) is not None}
    if mode == "train" or mode == "test":
        filtered_input_path = []
        filtered_eval_path = []
        for eval_p in eval_path:
            eval_vx_vy = extract_velocities(eval_p)
            if eval_vx_vy in input_dict:
                filtered_input_path.append(input_dict[eval_vx_vy])
                filtered_eval_path.append(eval_p)
    elif mode == "eval":
        filtered_input_path = input_path
        filtered_eval_path = eval_path
    all_file_names = [os.path.basename(path) for path in filtered_input_path]
    input_path = filtered_input_path
    eval_path = filtered_eval_path
    print(f"{len(input_path)=}")
    print(f"{len(eval_path)=}")
    num_cores = os.cpu_count()
    batch_size = 5000
    train_list,eval_list,file_names_list,vx_lists,vy_lists,inlet_left_lists,inlet_right_lists,outlet_left_lists,outlet_right_lists = [],[],[],[],[],[],[],[],[]
    #train_list,eval_list,file_names_list = [],[],[]
    for i in range(0, len(input_path),batch_size):
        #file_names = file_name_maker(f"{inputname}")
        #file_names = sort_and_combine_strings(file_names)
        batch_files = input_path[i:i+batch_size]
        batch_eval_files = eval_path[i:i+batch_size]
        file_names = all_file_names[i:i+batch_size]
        results = []
        if mode == "train" or mode == "test":
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                #results = list(tqdm(executor.map(process_file_mix, input_path,eval_path,[cut]*len(input_path)),total=len(input_path)))
                results = list(tqdm(executor.map(process_file_mix, batch_files, batch_eval_files, [cut]*len(batch_files)), total=len(batch_files)))
        elif mode == "eval":
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                #results = list(tqdm(executor.map(process_file_mix, input_path,eval_path,[cut]*len(input_path)),total=len(input_path)))
                results = list(tqdm(executor.map(process_file_mix, batch_files, batch_files, [cut]*len(batch_files)), total=len(batch_files)))
        train,evals = zip(*results)
        #形状の違うデータのファイルネームも先に削除
        train = list(train)
        evals = list(evals) 
        train = np.asarray(train, dtype=float)
        evals = np.asarray(evals, dtype=float)
        for j in range(len(train) - 1, -1, -1):  # 後ろから順に調べる
            if mode == "train" or mode == "test":
                if np.shape(train[j]) != (256, 2) or np.shape(evals[j]) != (1024, 2):
                #if np.shape(train[i]) != (1024, 2): mix_learningの時
                    print(f"del: {file_names[j]}")
                    del train[j]
                    del evals[j]
                    del file_names[j]
            elif mode == "eval":
                if np.shape(train[j]) != (256, 2):
                #if np.shape(train[i]) != (1024, 2): mix_learningの時
                    print(f"del: {file_names[j]}")
                    del train[j]
                    del evals[j]
                    del file_names[j]
        train = np.reshape(train, [len(train),16*16,2]) #高解像度の時
        #train = np.reshape(train, [len(train),width*width,2])
        if mode == "test" or mode=="train":
            evals = np.reshape(evals, [len(evals),width*width,2])
        elif mode == "eval":
            evals = np.reshape(evals, [len(evals),16*16,2])
        
        #delete nan file
       #get_file_name and delete nan file

        nan_indices = np.where(np.any(np.isnan(train),axis=(1,2)))[0]
        greater_than_two = np.any((train >= 20)|(train<=-20), axis=(1, 2))
        indices_to_remove = np.where(greater_than_two)[0]
        indices_to_remove = np.concatenate([indices_to_remove,nan_indices])
        indices_to_remove = np.sort(indices_to_remove)
        file_names = np.delete(file_names, indices_to_remove)
        train = np.delete(train, indices_to_remove, axis=0)
        evals = np.delete(evals, indices_to_remove, axis=0) 

        nan_indices = np.where(np.any(np.isnan(evals),axis=(1,2)))[0]
        #greater_than_two = np.any(evals >= 20, axis=(1, 2))
        greater_than_two = np.any((evals >= 20)|(evals<=-20), axis=(1, 2))
        indices_to_remove = np.where(greater_than_two)[0]
        indices_to_remove = np.concatenate([indices_to_remove,nan_indices])
        indices_to_remove = np.sort(indices_to_remove)

       #get_file_name and delete nan file
        file_names = np.delete(file_names, indices_to_remove)
        train = np.delete(train, indices_to_remove, axis=0)
        evals = np.delete(evals, indices_to_remove, axis=0) 
        # 正規表現を使用してvxとvyを抽出
        vx_list = []
        vy_list = []
        inlet_left_list = []
        inlet_right_list = []
        outlet_left_list = []
        outlet_right_list = []
        #velocity_x=-0.463_velocity_y=0.796_inlet_27_37_outlet_43_53
        for filename in file_names:
            vx_match = re.search(r"velocity_x=([-0-9.]+)", filename)
            vy_match = re.search(r"velocity_y=([-0-9.]+)", filename)

            inlet_match = re.search(r"inlet_(\d+)_(\d+)", filename)
            outlet_match = re.search(r"outlet_(\d+)_(\d+)", filename)

            if vx_match and vy_match:
                vx = float(vx_match.group(1))
                vy = float(vy_match.group(1))
            else:
                print("速度を抽出できませんでした") 
                sys.exit()
            vx_list.append(vx)
            vy_list.append(vy)
            
            if inlet_match and outlet_match:
            #if inlet_left_match and outlet_left_match:
                inlet_left = int(inlet_match.group(1))
                inlet_right = int(inlet_match.group(2))
                #inletをinlet_min,inlet_maxで正規化
                inlet_left = (inlet_left - inlet_min) / (inlet_max - inlet_min)
                inlet_right = (inlet_right - inlet_min) / (inlet_max - inlet_min)
                outlet_left = int(outlet_match.group(1))
                outlet_right = int(outlet_match.group(2))
            else:
                print("inletまたはoutletの値が見つかりませんでした")
                sys.exit()
            inlet_left_list.append(inlet_left)
            inlet_right_list.append(inlet_right)
            outlet_left_list.append(outlet_left)
            outlet_right_list.append(outlet_right)

        # 0.4以上の値を持つインデックスを見つけて削除
        indices_to_remove = [i for i, value in enumerate(vy_list) if value < 0.4]
        # indices_to_removeを逆順にして、vyとvxから要素を削除
        # 逆順にしないと、削除する際にインデックスがずれる可能性がある
        for index in sorted(indices_to_remove, reverse=True):
            del vy_list[index]
            del vx_list[index]
            del inlet_left_list[index]
            del inlet_right_list[index]
            del outlet_left_list[index]
            del outlet_right_list[index]
        file_names = np.delete(file_names, indices_to_remove)    
        train = np.delete(train, indices_to_remove, axis=0)
        evals = np.delete(evals, indices_to_remove, axis=0) 
        file_names = file_names.tolist()
        train = np.reshape(train, [-1,16,16,2]).transpose(0,3,1, 2) #高解像度の時
        if mode =="test" or mode=="train":
            evals = np.reshape(evals, [-1,width,width,2]).transpose(0,3,1,2)
        elif mode == "eval":
            evals = np.reshape(evals, [-1,16,16,2]).transpose(0,3,1,2)
        train = train[:len(evals)]
        file_names = file_names[:len(evals)]
        train_list.extend(train)
        eval_list.extend(evals)
        file_names_list.extend(file_names)
        vx_lists.extend(vx_list)
        vy_lists.extend(vy_list)
        inlet_left_lists.extend(inlet_left_list)
        inlet_right_lists.extend(inlet_right_list)
        outlet_left_lists.extend(outlet_left_list)
        outlet_right_lists.extend(outlet_right_list)
    avg_list = 0
    std_list = 0
    #return train_list, eval_list,file_names_list,avg_list,std_list,vx_list,vy_list,inlet_left_list,inlet_right_list,outlet_left_list,outlet_right_list
    return train_list,eval_list,file_names_list,avg_list,std_list,vx_lists,vy_lists,inlet_left_lists,inlet_right_lists,outlet_left_lists,outlet_right_lists

def Preprocessing_standard_mix_for(inputname,input_evalname,width,standard=0,cut=0,v2=False,mode="train"): #for文で複数ファイルをconcat
    pattarn_list = [12]
    for count,i in enumerate(pattarn_list):
        if mode == "train":
            train_file_path = f"{inputname}/inlet_size{i}_120000/Time=99" 
            eval_file_path = f"{input_evalname}/inlet_size{i}_120000/Time=99"
        else:
            train_file_path = f"{inputname}/inlet_size{i}_120000_test/Time=99" 
            eval_file_path = f"{input_evalname}/inlet_size{i}_120000_test/Time=99"

        if not os.path.exists(train_file_path) :
            print(train_file_path)
            print("訓練ファイルが存在しません")
            sys.exit()
        if not os.path.exists(eval_file_path) :        
            print(eval_file_path)
            print("評価ファイルが存在しません")
            sys.exit()

        if count == 0:
            train, evals,file_names,avg_list,std_list,vx_list,vy_list,inlet_left_list,inlet_right_list,outlet_left_list,outlet_right_list = sort_file_mix_for(train_file_path,eval_file_path,width,standard,cut)
        else:
            train2, evals2,file_names2,avg_list2,std_list2,vx_list2,vy_list2,inlet_left_list2,inlet_right_list2,outlet_left_list2,outlet_right_list2 = sort_file_mix_for(train_file_path,eval_file_path,width,standard)
            train = np.concatenate([train,train2],axis=0)
            evals = np.concatenate([evals,evals2],axis=0)
            file_names = np.concatenate([file_names,file_names2],axis=0)
            vx_list = np.concatenate([vx_list,vx_list2],axis=0)
            vy_list = np.concatenate([vy_list,vy_list2],axis=0)
            inlet_left_list = np.concatenate([inlet_left_list,inlet_left_list2],axis=0)
            inlet_right_list = np.concatenate([inlet_right_list,inlet_right_list2],axis=0)
            outlet_left_list = np.concatenate([outlet_left_list,outlet_left_list2],axis=0)
            outlet_right_list = np.concatenate([outlet_right_list,outlet_right_list2],axis=0)
        print(f"{len(train)=}")
        print(f"{len(evals)=}")
        print(f"{len(file_names)=}")
        print(f"{len(vx_list)=}")
        print(f"{len(vy_list)=}")
        print(f"{len(inlet_left_list)=}")
        print(f"{len(inlet_right_list)=}")
        print(f"{len(outlet_left_list)=}")
        print(f"{len(outlet_right_list)=}")
    return train, evals,file_names,avg_list,std_list,vx_list,vy_list,inlet_left_list,inlet_right_list,outlet_left_list,outlet_right_list

def Preprocessing_standard_mix(inputname,input_evalname,inputname2,input_evalname2,width,standard=0,cut=0,v2=False): #入り口の数が2つなら2v=True
    train, evals,file_names,avg_list,std_list,vx_list,vy_list,vx2_list,vy2_list = sort_file(inputname,input_evalname,width,standard,cut)
    train2, evals2,file_names2,avg_list2,std_list2,vx_list2,vy_list2,vx2_list2,vy2_list2 = sort_file(inputname2,input_evalname2,width,standard,cut)
    train = np.concatenate([train,train2],axis=0)
    evals = np.concatenate([evals,evals2],axis=0)
    file_names = np.concatenate([file_names,file_names2],axis=0)
    vx_list = np.concatenate([vx_list,vx_list2],axis=0)
    vy_list = np.concatenate([vy_list,vy_list2],axis=0)
    vx2_list = np.concatenate([vx2_list,vx2_list2],axis=0)
    return train, evals,file_names,avg_list,std_list,vx_list,vy_list,vx2_list,vy2_list

def Preprocessing_high_resolution(inputname,input_evalname,width,standard=0,cut=0,v2=False,mode="train"): #for文で複数ファイルをconcat
    size = 12
    if mode == "train":
        print("train")
        train_file_path = f"/mnt/data1/tony/inlet_value2/inlet_size{size}_120000/Time=99" 
        eval_file_path = f"/mnt/data1/tony/inlet_value1/inlet_size{size}_120000/Time=99" 
    else:
        train_file_path = f"/mnt/data1/tony/inlet_value2/inlet_size{size}_120000_test/Time=99" 
        eval_file_path = f"/mnt/data1/tony/inlet_value1/inlet_size{size}_120000_test/Time=99"

    if not os.path.exists(train_file_path) :
        print(train_file_path)
        print("訓練ファイルが存在しません")
        sys.exit()
    train, evals,file_names,avg_list,std_list,vx_list,vy_list,inlet_left_list,inlet_right_list,outlet_left_list,outlet_right_list = sort_file_high_resolution(train_file_path,eval_file_path,width,standard,cut,mode)
    print(f"{len(train)=}")
    print(f"{len(evals)=}")
    print(f"{len(file_names)=}")
    train = np.asarray(train)
    evals = np.asarray(evals)
    return train, evals,file_names,avg_list,std_list,vx_list,vy_list,inlet_left_list,inlet_right_list,outlet_left_list,outlet_right_list
