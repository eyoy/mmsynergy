import pandas as pd
import numpy as np
import json
import os
from itertools import groupby

motion_path = '/mnt/g/mmsynergy/alphapose_2d_norm/'
motions = [motion_path+i for i in os.listdir(motion_path)]
print(len(motions))

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

save_path = '/mnt/g/mmsynergy/landmark_features/'
for m in motions:
    with open(m, 'r') as f:
        df = json.load(f)
    id = m.split('/')[-1].split('_')[0]
    print(id)
    if os.path.exists(save_path +'{}_dist_angle.csv'.format(id)):
        print("Already done!")
    else:
        try:
            newlist = [list(g) for (k,g) in groupby(df, lambda item:item['image_id'])]
            newlist = [l[0] if len(l) == 1 else l[0] for l in newlist]
            results = pd.DataFrame()
            for i in newlist:
                name = i['image_id']
                frames = int(i['image_id'].split('.png')[0].split('_')[-1])
                frames_to_sec = frames/25
                keypoints = np.array(i['keypoints']).reshape(136,3)
                keypoints = np.delete(keypoints, 2,1)
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                shoulder_dist = np.sqrt(np.sum(np.square(left_shoulder - right_shoulder)))
                left_elbow = keypoints[7]
                right_elbow = keypoints[8]
                elbow_dist = np.sqrt(np.sum(np.square(left_elbow - right_elbow)))
                left_hand = keypoints[9]
                right_hand = keypoints[10]
                hand_dist = np.sqrt(np.sum(np.square(left_hand - right_hand)))
                left_knee = keypoints[13]
                right_knee = keypoints[14]
                knee_dist = np.sqrt(np.sum(np.square(left_knee - right_knee)))
                left_ankle = keypoints[15]
                right_ankle = keypoints[16]
                ankle_dist = np.sqrt(np.sum(np.square(left_ankle - right_ankle)))
                head = keypoints[17]
                hip = keypoints[19]
                l_elbow_angle = calculate_angle(left_shoulder,left_elbow, left_hand)
                r_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_hand)
                l_knee_angle = calculate_angle(hip, left_knee, left_ankle)
                r_knee_angle = calculate_angle(hip, right_knee, right_ankle)
                l_hip_angle = calculate_angle(left_shoulder, hip, left_knee)
                r_hip_angle = calculate_angle(right_shoulder,hip,right_knee)
                d = {'img_id': [name], 'frame' : [frames], 'sec':[frames_to_sec], 'shoulder_dist': [shoulder_dist], "elbow_dist":[elbow_dist], "hand_dist":[hand_dist], "knee_dist":[knee_dist],
                    "ankle_dist":[ankle_dist],'head':[head],'hip':[hip],'left_shoulder': [left_shoulder], "right_shoulder":[right_shoulder], 
                    "left_elbow":[left_elbow], "right_elbow":[right_elbow],"left_hand":[left_hand],"right_hand":[right_hand], 
                    "left_knee":[left_knee],"right_knee":[right_knee],"left_ankle":[left_ankle],"right_ankle":[right_ankle],
                    'l_elbow_angle':[l_elbow_angle], "r_elbow_angle":[r_elbow_angle], "l_knee_angle":[l_knee_angle],"r_knee_angle":[r_knee_angle], 
                    "l_hip_angle":[l_hip_angle],"r_hip_angle":[r_hip_angle]}
                df = pd.DataFrame(d)
                results = pd.concat([results, df], axis=0).reset_index(drop=True)
            COL = ['head', 'hip', 'left_shoulder','right_shoulder', 'left_elbow', 'right_elbow', 'left_hand',
            'right_hand', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
            sec_per_frame = results.sec[1]-results.sec[0]
            dist_by_frame = {}
            for c in COL:
                dist_by_frame['{}'.format(c)] = []
                for i,j in zip(list(results[c]),list(results[c].shift(1))):
                    if j is not None:
                        head = np.linalg.norm(i-j)
                        dist_by_frame['{}'.format(c)].append(head)
            df_dist = pd.DataFrame(dist_by_frame)
            for c in df_dist.columns:
                df_dist[c+'_velocity'] = df_dist[c]/sec_per_frame
            results.to_csv(save_path +'{}_dist_angle.csv'.format(id))
            print("Angle results saved!")
            df_dist.to_csv(save_path + "{}_velocity.csv".format(id))
            print("velocity reuslts saved!")
        except:
            print("Error")
            continue
            

