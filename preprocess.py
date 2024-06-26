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

def process_file(path, eval_path,cut=0):
    df = pd.read_csv(path)
    df = df.values
    df = df.astype(float)
    #df配列の中で0.5以下の要素を0に変換
    df = np.where(df <= cut, 0, df)
    df_eval = pd.read_csv(eval_path)
    df_eval = df_eval.values
    df_eval = df_eval.astype(float)
    df_eval = np.where(df_eval <= cut, 0, df_eval)
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

def Preprocessing_standard(inputname,input_evalname,width,standard=0,cut=0):
    input_path = sorted(glob.glob(inputname + "/*.csv"), key=natural_keys)
    eval_path = sorted(glob.glob(input_evalname + "/*.csv"), key=natural_keys)
    #print(f"before_{len(input_path)=}")
    #print(f"before_{len(eval_path)=}")
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

    #print(f"delete_index={indices_to_remove}")
    #print(f"len(delete_index)={len(indices_to_remove)}")
    #print(f"after_{len(train)=}")
    #sys.exit()


    #get_file_name and delete nan file
    file_names = file_name_maker(f"{inputname}")
    file_names = sort_and_combine_strings(file_names)
    file_names = np.delete(file_names, indices_to_remove)
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
    return train, evals,file_names,avg_list,std_list
