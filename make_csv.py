import os
import shutil
import subprocess
import numpy as np
import csv 
import sys
import glob
from tqdm import tqdm
import pandas as pd
import datetime
import pytz

#指定した時間ステップの速度データを抽出し、csvファイルに保存する
def extract_velocity(case_directory, time_steps):
    for time_step in time_steps:
        velocities = []
        u_file_path = os.path.join(case_directory, str(time_step), "U")
        #ファイルが存在しない場合は0を返す
        if not os.path.exists(u_file_path):
            return 0
        with open(u_file_path, 'r') as file:
            lines = file.readlines()
        start_reading = False
        for line in lines:
            if line.startswith("internalField"):
                start_reading = True
                continue
            if start_reading:
                if line.startswith("(") and ")" in line:
                    velocity_vector = line.strip().strip("()").split()
                    x_velocity, y_velocity = float(velocity_vector[0]), float(velocity_vector[1])
                    velocities.append([x_velocity,y_velocity])
        with open(f"{case_directory}/velocity_x={velocity_x:.3f}_velocity_y={velocity_y:.3f}_time={time_step}.csv", 'w', newline='') as file:
            writer = csv.writer(file)                  
            writer.writerow(['X Velocity',"Y Velocity"])
            writer.writerows(velocities)
    return np.array(velocities)

case_directory = "../train_data_ver6/results"
decimal = 3
time_steps = [5]
#initial_velocities_x = np.linspace(-0.8, 0.8, 200)
#initial_velocities_y = np.linspace(0.8, 0.1, 100)
initial_velocities_x = np.linspace(0.8, 0.1, 25)
initial_velocities_y = np.linspace(0.8, 0.1, 40)
initial_velocities_x = np.around(initial_velocities_x, decimals=decimal) 
initial_velocities_y = np.around(initial_velocities_y, decimals=decimal)


for velocity_x in tqdm(initial_velocities_x, total=len(initial_velocities_x)):
    for velocity_y in initial_velocities_y:
        _ = extract_velocity(f"{case_directory}/velocity_x={velocity_x:.3f}_velocity_y={velocity_y:.3f}", time_steps)
