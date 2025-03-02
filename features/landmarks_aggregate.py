import pandas as pd
import numpy as np
import json
import os

input_path = '/mnt/g/mmsynergy/landmark_features/'
files = [input_path+f for f in os.listdir(input_path)]
print(len(files))

#with open('/large_data_storage3/erya/final_ids.json', 'r') as f:
    #ids = json.load(f)
d = pd.read_csv('/mnt/g/mmsynergy/regression/mult_mtl_contextual_reg_merged.csv')
d.id = d.id.astype(str)
ids = d.id.to_list()

results = pd.DataFrame()
for i in ids:
    print(i, ids.index(i))
    f = '/mnt/g/mmsynergy/landmark_features/{}_dist_angle.csv'.format(i)
    v = '/mnt/g/mmsynergy/landmark_features/{}_velocity.csv'.format(i)
    try:
        dist = pd.read_csv(f, index_col = 0)
        COL = ['shoulder_dist', 'elbow_dist', 'hand_dist','knee_dist', 'ankle_dist','l_elbow_angle', 'r_elbow_angle', 'l_knee_angle', 'r_knee_angle',
            'l_hip_angle', 'r_hip_angle']
        dist = dist[COL]
        df_mean = dist.mean().to_frame().T
        df_mean = df_mean.add_suffix('_mean')
        df_var = dist.var().to_frame().T
        df_var = df_var.add_suffix('_var')
        dist_angle = pd.concat([df_mean, df_var], axis = 1)
        dist_angle['id'] = f.split('/')[-1].split('_')[0]
        velocity = pd.read_csv(v, index_col = 0)
        v_mean = velocity.mean().to_frame().T
        v_mean = v_mean.add_suffix('_mean')
        v_var = velocity.var().to_frame().T
        v_var = v_var.add_suffix('_var')
        vl = pd.concat([v_mean,v_var], axis=1)
        #vl['id'] = v.split('/')[-1].split('_')[0]
        total = pd.concat([dist_angle,vl], axis=1)
        results = pd.concat([results, total], axis=0).reset_index(drop=True)
    except:
        continue

results.to_csv("/mnt/g/mmsynergy/regression/motion_total.csv")
print("Successfully saved!")