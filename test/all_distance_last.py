import numpy as np
import pandas as pd
import os
from scipy.spatial import distance
from scipy import stats
from dtw import *
# env multibench
data_path = '/mnt/g/mmsynergy/matrix'
files = [os.path.join(data_path, i) for i in os.listdir(data_path)]
ids = []
dtw_b = []
braycurtis_b = []
canberra_b = []
chebyshev_b = []
manhattan_b = []
correlation_b = []
cosine_b = []
euclidean_b = []
minkowski_b = []
sqeuclidean_b = []
pearson_b = []
spearman_b = []
for f in files:
    id = f.split('/')[-1].split('.npz')[0]
    data = np.load(f)
    mat1 = data['arr_0'][0][0]
    mat2 = data['arr_0'][0][1]
    dtw_dist = dtw(mat1, mat2, keep_internals=True,dist_method='euclidean', distance_only=True).distance
    dtw_b.append(dtw_dist)
    v_mean = mat1[-1]
    a_mean = mat2[-1]
    braycurtis = distance.braycurtis(v_mean, a_mean)
    canberra = distance.canberra(v_mean,a_mean)
    chebyshev = distance.chebyshev(v_mean,a_mean)
    manhattan = distance.cityblock(v_mean,a_mean)
    correlation = distance.correlation(v_mean,a_mean)
    cosine = distance.cosine(v_mean, a_mean)
    euclidean = distance.euclidean(v_mean, a_mean)
    minkowski = distance.minkowski(v_mean, a_mean)
    sqeuclidean = distance.sqeuclidean(v_mean, a_mean)
    pearson = stats.pearsonr(v_mean, a_mean)[0]
    spearman = stats.spearmanr(v_mean, a_mean)[0]
    ids.append(id)
    braycurtis_b.append(braycurtis)
    canberra_b.append(canberra)
    chebyshev_b.append(chebyshev)
    manhattan_b.append(manhattan)
    correlation_b.append(correlation)
    cosine_b.append(cosine)
    euclidean_b.append(euclidean)
    minkowski_b.append(minkowski)
    sqeuclidean_b.append(sqeuclidean)
    pearson_b.append(pearson)
    spearman_b.append(spearman)
    
df = pd.DataFrame(list(zip(ids, dtw_b,braycurtis_b, canberra_b,chebyshev_b,manhattan_b,correlation_b,cosine_b,euclidean_b,minkowski_b,sqeuclidean_b, pearson_b, spearman_b)),
               columns =['id', 'dtw','braycurtis', 'canberra','chebyshev','manhattan','correlation','cosine','euclidean','minkowski','sqeuclidean','pearson','spearman'])
df.to_csv('/mnt/g/mmsynergy/regression/all_dist_mtl_last.csv')
print("The length of data is " + str(len(data)))
print("Saved!")