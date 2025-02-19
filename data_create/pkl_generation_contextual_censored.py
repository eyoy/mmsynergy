# env multibench
import json
import numpy as np
import os
import glob
import pandas as pd
from collections import Counter
import pickle
# 20250205 add structured data such as genre as contextual layer, not just video description
import json
from sklearn.preprocessing import OneHotEncoder

audio_f_path = '/mnt/g/mmsynergy/audio_features/'
video_f_path = '/mnt/g/mmsynergy/alphapose_2d_norm/'
text_f_path = '/mnt/g/mmsynergy/bert/'
audio_json  = [audio_f_path+a for a in os.listdir(audio_f_path)]


clean_ids = [i.split('.mp4')[0] for i in os.listdir('/mnt/g/mmsynergy/single_videos_cleaned/') if i.endswith('.mp4')]
print(len(clean_ids))
# all_ids = visual_id_train + visual_id_valid + visual_id_test
# print(len(all_ids))
video = pd.read_csv('/mnt/g/mmsynergy/singles_meta_cleaned.csv')
video['id'] = video['id'].astype(str)
video = video[~video.text.isnull()]
tutorial_id = video.loc[video["text"].str.contains("tutorial"),'id'].to_list()
clean_ids = [i for i in clean_ids if i not in tutorial_id]
print(len(clean_ids))
beats = pd.read_csv('/mnt/g/mmsynergy/music_beats_duration.csv', index_col=0)
beats['id'] = beats['id'].astype(str)
beats = beats[beats.id.isin(clean_ids)]
beats = beats[beats.beats_per_video <= 50]
max_beats = int(beats['beats_per_video'].max())
print(max_beats)
#video = pd.read_csv('/mnt/f/erya/mmsynergy/singles_meta_cleaned.csv')
#video['id'] = video['id'].astype(str)
single = video[video.id.isin(beats.id)]
single.drop('duration', axis=1, inplace=True)
print(single.shape)
d = pd.read_csv('/mnt/g/mmsynergy/regression/mult_mtl_contextual_reg_merged.csv')
d['id'] = d['id'].astype(str)
single = single.merge(d[['id', 'follower', 'following', 'like',
       'video', 'genre', 'bpm', 'beats_per_video', 'gender', 'race', 'emo',
       'age', 'create_date', 'create_day','duration']], on='id', how = 'left')
single = single[single.gender.notnull()]
single[['follower', 'following', 'like', 'video']] = single[['follower', 'following', 'like', 'video']].replace(np.nan,0)

print(single.shape)


# visual_id_train = [i for i in visual_id_train if i in clean_ids]
# print(len(visual_id_train))
# visual_id_valid = [i for i in visual_id_valid if i in clean_ids]
# print(len(visual_id_valid))
# visual_id_test = [i for i in visual_id_test if i in clean_ids]
# print(len(visual_id_test))
from sklearn.model_selection import train_test_split

visual_id_train, visual_id_remaining = train_test_split(list(single.id), test_size=0.6, random_state=44) #40757 16302 24455
#split the test again to get 20% dev and 20% tests
visual_id_valid,visual_id_test = train_test_split(visual_id_remaining, test_size=0.5, random_state=44)
print(len(visual_id_train), len(visual_id_valid), len(visual_id_test))
print(len(visual_id_train)+len(visual_id_valid)+len(visual_id_test))
with open ('/mnt/g/mmsynergy/model/train_id_contextual.json', 'w') as f:
    json.dump(visual_id_train, f)
with open ('/mnt/g/mmsynergy/model/valid_id_contextual.json', 'w') as f:
    json.dump(visual_id_valid, f)
with open ('/mnt/g/mmsynergy/model/test_id_contextual.json', 'w') as f:
    json.dump(visual_id_test, f)



p = []
audio_box_train = []
for n in visual_id_train:
    path_name = audio_f_path + '{}.json'.format(n)
    p.append(path_name)
    f = open(path_name)
  
    # returns JSON object as a dictionary
    df = json.load(f)
    df = np.array(df[0])
    audio_box_train.append(df)

# find the max # of beats
#n_beats = [d.shape[0] for d in audio_box]
#beats_max = max(n_beats)

#padding
data_audio_train = []
for df in audio_box_train:
    if df.shape[0] < max_beats:
        df = np.pad(df, [(0, max_beats - len(df)%max_beats), (0,0)], 'constant')
    else:
        df = df
    data_audio_train.append(df)

#reshape the audio data
audio_train = np.dstack(data_audio_train)
audio_train = np.rollaxis(audio_train,-1)
    
from itertools import groupby
# append all visual features together
#name_box = []
visual_box_train = []
for i in visual_id_train:
    print(i, visual_id_train.index(i))
    if os.path.exists(video_f_path+i+'_alphapose-results-norm.json'):
        print(str(video_f_path+i+'_alphapose-results-norm.json'))
        f = open(video_f_path+i+'_alphapose-results-norm.json')
    # returns JSON object as a dictionary
    df = json.load(f)
    newlist = [list(g) for (k,g) in groupby(df, lambda item:item['image_id'])]
    newlist = [l[0] if len(l) == 1 else l[0] for l in newlist]
    landmarks = []
    ids = []
    for ll in newlist:
        keypoints = np.array(ll['keypoints']).reshape(136,3)
        keypoints = np.delete(keypoints, 2,1)
        keypoints = keypoints.reshape(272,)
        landmarks.append(keypoints)
    if np.array(landmarks).shape[0] < max_beats:
        df = np.pad(np.array(landmarks), [(0, max_beats - len(landmarks)%max_beats), (0,0)], 'constant')
    else:
        df = np.array(landmarks)
    print(df.shape)
    visual_box_train.append(np.array(df))
    
# find the max # of beats
#max_beats = max([int(vis.split('.png')[0].split('_')[-1]) for vis in visual_json[:20]])

#padding
# data_visual_train = []
# for df in visual_box_train:
#     if df.shape[0] < max_beats:
#         df = np.pad(df, [(0, max_beats - len(df)%max_beats), (0,0)], 'constant')
#     else:
#         df = df
#     data_visual_train.append(df)
    
#reshape the audio data
visual_train = np.dstack(visual_box_train)
visual_train = np.rollaxis(visual_train,-1)
print(visual_train.shape)
# get contextual
text_box_train = []
for i in visual_id_train:
    with open(text_f_path+i+'.json') as f:
        text = json.load(f)
        text_box_train.append(text)
text_train = np.array(text_box_train) #N x 768
print(text_train.shape)
encoder = OneHotEncoder()
d = single[single.id.isin(visual_id_train)]
genre = encoder.fit_transform(d[['genre']]).toarray() # N x 7
genre = genre.reshape(genre.shape[0], 1, genre.shape[1])
print(genre.shape)
race = encoder.fit_transform(d[['race']]).toarray() # N x 6
race = race.reshape(race.shape[0], 1, race.shape[1])
print(race.shape)
gender = encoder.fit_transform(d[['gender']]).toarray() # N x 2
gender = gender.reshape(gender.shape[0], 1, gender.shape[1])
print(gender.shape)
d['create_date'] = pd.to_datetime(d['create_date'])
d['timestamp'] = d['create_date'].astype('int64') / 10**9
continuous = d[['age','duration']].to_numpy() # N x 2
continuous = continuous.reshape(continuous.shape[0], 1, continuous.shape[1])
print(continuous.shape)
contextual_train = np.concatenate((text_train, genre, race, gender, continuous), axis = 2)
print(contextual_train.shape)
#get outcome variables

names_df = pd.DataFrame(visual_id_train).reset_index()
names_df = names_df.rename(columns = {'index':'order_name',0:'id' })
dance = single.merge(names_df, on = 'id', how = 'inner')
dance = dance.sort_values('order_name')
print(dance.shape)
dance['digg_count'] = np.log(dance['digg_count']+1)
dance['share_count'] = np.log(dance['share_count']+1)
dance['comment_count'] = np.log(dance['comment_count']+1)
dance['play_count'] = np.log(dance['play_count']+1)
digg_train = np.array(dance['digg_count']).reshape(len(dance['digg_count']),1,1)
share_train = np.array(dance['share_count']).reshape(len(dance['share_count']),1,1)
comment_train = np.array(dance['comment_count']).reshape(len(dance['comment_count']),1,1)
play_train = np.array(dance['play_count']).reshape(len(dance['play_count']),1,1)

#shallow_train = np.array(dance['shallow_log']).reshape(len(dance['shallow_log']),1,1)
#deep_train = np.array(dance['deep_log']).reshape(len(dance['deep_log']),1,1)


#get id
ids_train = np.array(visual_id_train).reshape(len(visual_id_train),1)

print('train data done!')
#data generated
#digg_tr = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_train, 'labels':digg_train}}
#share_tr = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_train, 'labels':share_train}}
#comment_tr = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_train, 'labels':comment_train}}
#play_tr = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_train, 'labels':play_train}}

    
### test data
audio_box_test = []
for n in visual_id_test:
    path_name = audio_f_path + '{}.json'.format(n)
    f = open(path_name)
  
    # returns JSON object as a dictionary
    df = json.load(f)
    df = np.array(df[0])
    audio_box_test.append(df)


#padding
data_audio_test = []
for df in audio_box_test:
    if df.shape[0] < max_beats:
        df = np.pad(df, [(0, max_beats - len(df)%max_beats), (0,0)], 'constant')
    else:
        df = df
    data_audio_test.append(df)

#reshape the audio data
audio_test = np.dstack(data_audio_test)
audio_test = np.rollaxis(audio_test,-1)
    
from itertools import groupby
# append all visual features together
#name_box = []
visual_box_test = []
for i in visual_id_test:
    print(i, visual_id_test.index(i))
    #name = i.split('/')[-1].split('_')[0]
    #name_box.append(name)
    
    #p_name = video_f_path + '{}/'.format(i)
    #path = p_name + '*.json'
    if os.path.exists(video_f_path+i+'_alphapose-results-norm.json'):
        print(str(video_f_path+i+'_alphapose-results-norm.json'))
        f = open(video_f_path+i+'_alphapose-results-norm.json')
  
    # returns JSON object as a dictionary
    df = json.load(f)
    newlist = [list(g) for (k,g) in groupby(df, lambda item:item['image_id'])]
    newlist = [l[0] if len(l) == 1 else l[0] for l in newlist]
    landmarks = []
    ids = []
    for ll in newlist:
        keypoints = np.array(ll['keypoints']).reshape(136,3)
        keypoints = np.delete(keypoints, 2,1)
        keypoints = keypoints.reshape(272,)
        landmarks.append(keypoints)
    if np.array(landmarks).shape[0] < max_beats:
        df = np.pad(np.array(landmarks), [(0, max_beats - len(landmarks)%max_beats), (0,0)], 'constant')
    else:
        df = np.array(landmarks)
    print(df.shape)
    visual_box_test.append(np.array(df))
    
# find the max # of beats
#max_beats = max([int(vis.split('.png')[0].split('_')[-1]) for vis in visual_json[:20]])

#padding
# data_visual_test = []
# for df in visual_box_test:
#     if df.shape[0] < max_beats:
#         df = np.pad(df, [(0, max_beats - len(df)%max_beats), (0,0)], 'constant')
#     else:
#         df = df
#     data_visual_test.append(df)
    
#reshape the audio data
visual_test = np.dstack(visual_box_test)
visual_test = np.rollaxis(visual_test,-1)

# get contextual
text_box_test = []
for i in visual_id_test:
    with open(text_f_path+i+'.json') as f:
        text = json.load(f)
        text_box_test.append(text)
text_test = np.array(text_box_test) #N x 768
encoder = OneHotEncoder()
d = single[single.id.isin(visual_id_test)]
genre = encoder.fit_transform(d[['genre']]).toarray() # N x 7
genre = genre.reshape(genre.shape[0], 1, genre.shape[1])
print(genre.shape)
race = encoder.fit_transform(d[['race']]).toarray() # N x 6
race = race.reshape(race.shape[0], 1, race.shape[1])
print(race.shape)
gender = encoder.fit_transform(d[['gender']]).toarray() # N x 2
gender = gender.reshape(gender.shape[0], 1, gender.shape[1])
print(gender.shape)
d['create_date'] = pd.to_datetime(d['create_date'])
d['timestamp'] = d['create_date'].astype('int64') / 10**9
continuous = d[['age','duration']].to_numpy() # N x 2
continuous = continuous.reshape(continuous.shape[0], 1, continuous.shape[1])
print(continuous.shape)
contextual_test = np.concatenate((text_test, genre, race, gender, continuous), axis = 2)
print(contextual_test.shape)

#get outcome variables
names_df = pd.DataFrame(visual_id_test).reset_index()
names_df = names_df.rename(columns = {'index':'order_name',0:'id' })
dance = single.merge(names_df, on = 'id', how = 'inner')
dance = dance.sort_values('order_name')
dance['digg_count'] = np.log(dance['digg_count']+1)
dance['share_count'] = np.log(dance['share_count']+1)
dance['comment_count'] = np.log(dance['comment_count']+1)
dance['play_count'] = np.log(dance['play_count']+1)
digg_test = np.array(dance['digg_count']).reshape(len(dance['digg_count']),1,1)
share_test = np.array(dance['share_count']).reshape(len(dance['share_count']),1,1)
comment_test = np.array(dance['comment_count']).reshape(len(dance['comment_count']),1,1)
play_test = np.array(dance['play_count']).reshape(len(dance['play_count']),1,1)
# shallow_test = np.array(dance['shallow_log']).reshape(len(dance['shallow_log']),1,1)
# deep_test = np.array(dance['deep_log']).reshape(len(dance['deep_log']),1,1)

#get id

ids_test = np.array(visual_id_test).reshape(len(visual_id_test),1)

#data generated
#digg_te = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_test, 'labels':digg_test}}
#share_te = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_train, 'labels':share_test}}
#comment_te = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_train, 'labels':comment_test}}
#play_te = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_train, 'labels':play_test}}

print('test part done!')
### valid data
audio_box_valid = []
for n in visual_id_valid:
    path_name = audio_f_path + '{}.json'.format(n)
    f = open(path_name)
  
    # returns JSON object as a dictionary
    df = json.load(f)
    df = np.array(df[0])
    audio_box_valid.append(df)


#padding
data_audio_valid = []
for df in audio_box_valid:
    if df.shape[0] < max_beats:
        df = np.pad(df, [(0, max_beats - len(df)%max_beats), (0,0)], 'constant')
    else:
        df = df
    data_audio_valid.append(df)

#reshape the audio data
audio_valid = np.dstack(data_audio_valid)
audio_valid = np.rollaxis(audio_valid,-1)
    
from itertools import groupby
# append all visual features together
#name_box = []
visual_box_valid = []
for i in visual_id_valid:
    print(i, visual_id_valid.index(i))
    #name = i.split('/')[-1].split('_')[0]
    #name_box.append(name)
    #p_name = video_f_path + '{}/'.format(i)
    #path = p_name + '*.json'
    if os.path.exists(video_f_path+i+'_alphapose-results-norm.json'):
        print(str(video_f_path+i+'_alphapose-results-norm.json'))
        f = open(video_f_path+i+'_alphapose-results-norm.json')
  
    # returns JSON object as a dictionary
    df = json.load(f)
    newlist = [list(g) for (k,g) in groupby(df, lambda item:item['image_id'])]
    newlist = [l[0] if len(l) == 1 else l[0] for l in newlist]
    landmarks = []
    ids = []
    for ll in newlist:
        keypoints = np.array(ll['keypoints']).reshape(136,3)
        keypoints = np.delete(keypoints, 2,1)
        keypoints = keypoints.reshape(272,)
        landmarks.append(keypoints)
    if np.array(landmarks).shape[0] < max_beats:
        df = np.pad(np.array(landmarks), [(0, max_beats - len(landmarks)%max_beats), (0,0)], 'constant')
    else:
        df = np.array(landmarks)
    print(df.shape)
    visual_box_valid.append(np.array(df))
    
# find the max # of beats
#max_beats = max([int(vis.split('.png')[0].split('_')[-1]) for vis in visual_json[:20]])

#padding
# data_visual_valid = []
# for df in visual_box_valid:
#     if df.shape[0] < max_beats:
#         df = np.pad(df, [(0, max_beats - len(df)%max_beats), (0,0)], 'constant')
#     else:
#         df = df
#     data_visual_valid.append(df)
    
#reshape the audio data
visual_valid = np.dstack(visual_box_valid)
visual_valid = np.rollaxis(visual_valid,-1)

# get text
text_box_valid = []
for i in visual_id_valid:
    with open(text_f_path+i+'.json') as f:
        text = json.load(f)
        text_box_valid.append(text)
text_valid = np.array(text_box_valid) #N x 768
encoder = OneHotEncoder()
d = single[single.id.isin(visual_id_valid)]
genre = encoder.fit_transform(d[['genre']]).toarray() # N x 7
genre = genre.reshape(genre.shape[0], 1, genre.shape[1])
print(genre.shape)
race = encoder.fit_transform(d[['race']]).toarray() # N x 6
race = race.reshape(race.shape[0], 1, race.shape[1])
print(race.shape)
gender = encoder.fit_transform(d[['gender']]).toarray() # N x 2s
gender = gender.reshape(gender.shape[0], 1, gender.shape[1])
print(gender.shape)
d['create_date'] = pd.to_datetime(d['create_date'])
d['timestamp'] = d['create_date'].astype('int64') / 10**9
continuous = d[['age','duration']].to_numpy() # N x 2
continuous = continuous.reshape(continuous.shape[0], 1, continuous.shape[1])
print(continuous.shape)
contextual_valid = np.concatenate((text_valid, genre, race, gender, continuous), axis = 2)

#get outcome variables

names_df = pd.DataFrame(visual_id_valid).reset_index()
names_df = names_df.rename(columns = {'index':'order_name',0:'id' })
dance = single.merge(names_df, on = 'id', how = 'inner')
dance = dance.sort_values('order_name')
dance['digg_count'] = np.log(dance['digg_count']+1)
dance['share_count'] = np.log(dance['share_count']+1)
dance['comment_count'] = np.log(dance['comment_count']+1)
dance['play_count'] = np.log(dance['play_count']+1)
digg_valid = np.array(dance['digg_count']).reshape(len(dance['digg_count']),1,1)
share_valid = np.array(dance['share_count']).reshape(len(dance['share_count']),1,1)
comment_valid = np.array(dance['comment_count']).reshape(len(dance['comment_count']),1,1)
play_valid = np.array(dance['play_count']).reshape(len(dance['play_count']),1,1)
# shallow_valid = np.array(dance['shallow_log']).reshape(len(dance['shallow_log']),1,1)
# deep_valid = np.array(dance['deep_log']).reshape(len(dance['deep_log']),1,1)

#get id

ids_valid = np.array(visual_id_valid).reshape(len(visual_id_valid),1)
print('valid part done')
#data generated
digg  = {'train':{'vision':visual_train, 'audio':audio_train, 'text': contextual_train, 'id':ids_train, 'labels':digg_train},
        'valid':{'vision':visual_valid, 'audio':audio_valid, 'text': contextual_valid,'id':ids_valid, 'labels':digg_valid},
        'test':{'vision':visual_test, 'audio':audio_test, 'text':contextual_test, 'id':ids_test, 'labels':digg_test}}
# share = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_train, 'labels':share_train},
#         'valid':{'vision':visual_valid, 'audio':audio_valid, 'id':ids_valid, 'labels':share_valid},
#         'test':{'vision':visual_test, 'audio':audio_test, 'id':ids_test, 'labels':share_test}}
comment = {'train':{'vision':visual_train, 'audio':audio_train, 'text': contextual_train, 'id':ids_train, 'labels':comment_train},
        'valid':{'vision':visual_valid, 'audio':audio_valid, 'text': contextual_valid, 'id':ids_valid, 'labels':comment_valid},
        'test':{'vision':visual_test, 'audio':audio_test, 'text':contextual_test, 'id':ids_test, 'labels':comment_test}}
# play = {'train':{'vision':visual_train, 'audio':audio_train, 'id':ids_train, 'labels':play_train},
#         'valid':{'vision':visual_valid, 'audio':audio_valid, 'id':ids_valid, 'labels':play_valid},
#         'test':{'vision':visual_test, 'audio':audio_test, 'id':ids_test, 'labels':play_test}}
mtl = {'train':{'vision':visual_train, 'audio':audio_train, 'text': contextual_train, 'id':ids_train, 'labels1':digg_train, 'labels2':comment_train,'label_p': play_train},
        'valid':{'vision':visual_valid, 'audio':audio_valid, 'text': contextual_valid, 'id':ids_valid, 'labels1':digg_valid, 'labels2':comment_valid, 'label_p': play_valid},
        'test':{'vision':visual_test, 'audio':audio_test, 'text':contextual_test, 'id':ids_test, 'labels1':digg_test, 'labels2':comment_test, 'label_p': play_test}}

with open('/mnt/g/mmsynergy/model/data/digg_contextual_censored.pkl', 'wb') as f:
    pickle.dump(digg, f)
print('/mnt/g/mmsynergy/model/data/digg_contextual_censored.pkl')
# with open('/mnt/e/erya/mult_rr_data/new_0115/share.pkl', 'wb') as f:
#     pickle.dump(share, f)
# print('/mnt/e/erya/mult_rr_data/new_0115/share.pkl')
with open('/mnt/g/mmsynergy/model/data/comment_contextual_censored.pkl', 'wb') as f:
    pickle.dump(comment, f)
print('/mnt/g/mmsynergy/model/data/comment_contextual_censored.pkl')
# with open('/mnt/e/erya/mult_rr_data/new_0115/play.pkl', 'wb') as f:
#     pickle.dump(play, f)
# print('/mnt/e/erya/mult_rr_data/new_0115/play.pkl')
with open('/mnt/g/mmsynergy/model/data/mtl_contextual_censored.pkl', 'wb') as f:
    pickle.dump(mtl, f)
print('/mnt/g/mmsynergy/model/data/mtl_contextual_censored.pkl')
