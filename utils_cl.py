import numpy as np
import pandas as pd
import random as rd
import math

from datetime import timedelta
from tqdm import tqdm_notebook
#from dtaidistance import dtw

def distance_ED(array1, array2) :
    # Caculate Euclidean distance for 2 time series samples
    
    array1 = list(array1)
    array2 = list(array2)
    def square(_list):
        return [_ele * _ele for _ele in _list]
    subtract_list = [a - b for (a, b) in zip(array1, array2)]
    
    return math.sqrt(sum(square(subtract_list)))


def distance_DTW(array1, array2) :
    # Caculate Dynamic Time Warping distance for 2 time series samples
    
    #array1 = list(array1)
    #array2 = list(array2)
    distance = dtw.distance(array1, array2)
    
    return distance


def extract_n_months(df, n, is_drop_missing = False):
    ## Extract feature : intervals of months for dataset
    # n : iterval of month - should be divisor of total months
    # drop_missing : is drop missing month sample
    
    def get_month(date_time):
        month = str(date_time.month)
        year = str(date_time.year)
        return pd.Timestamp(year + '-' + month)
    
    df['MONTH'] = df['TIME'].apply(get_month)
    
    mean_consumption = df.groupby(["MA_DIEMDO", "MONTH"])['CONSUMPTION'].mean().reset_index()
    list_diemdo = list(set(mean_consumption['MA_DIEMDO']))
    
    dict_df = {}
    dict_df['MA_DIEMDO'] = list_diemdo
    first_dd = list(mean_consumption.loc[mean_consumption['MA_DIEMDO'] == list_diemdo[0]]['CONSUMPTION'])
    lack_mdd = []
    num_interval = 15
    # Get rid of 4 last months ( of 2018 )
    
    nameTime_len = 7
    
    mdd = []
    for i in range(num_interval):
        res_value = []
        for _mdd in list_diemdo :
            try :
                res_value.append(list(mean_consumption.loc[mean_consumption['MA_DIEMDO'] == _mdd]['CONSUMPTION'])[i])
            except :
                lack_mdd.append(_mdd)
                res_value.append(0)
                # Treat missing value as 0
        dict_df[str(mean_consumption['MONTH'][i])[:nameTime_len]] = res_value

    res_df = pd.DataFrame(dict_df)
    if is_drop_missing:
        for _mdd in list(set(lack_mdd)):
            res_df.drop(res_df[res_df.MA_DIEMDO == _mdd].index, inplace=True)
            list_diemdo.remove(_mdd)
    res_df.reset_index(inplace = True)
    
    return res_df, list_diemdo

from tqdm import tqdm

def K_Mean(X, n_iters, cluster_number, metric = 'euclidean'):
    ## K_Mean for hard - clustering algorithm
    
    if metric == 'euclidean':
        distance_used = distance_ED
    elif metric == 'dtw':
        distance_used = distance_DTW
    m = X.shape[0] # number of samples
    n = X.shape[1] # number of monthly features
    centroid = np.array([]).reshape(n, 0)
    for i in range(cluster_number):
        centroid = np.c_[centroid, X[rd.randint(0, m-1)]]
    # Centroid is a n x cluster_number dimentional matrix, where each column will be a centroid for one cluster
    
    # Assign points to each clusters (based on ED distance)
    Y = {}
    for i in tqdm(range(n_iters)):
        
        EuclidianDistance = np.array([]).reshape(m,0)
        for j in range(cluster_number):
            tempDist = np.sum((X - centroid[:, j])**2,axis=1)
            EuclidianDistance = np.c_[EuclidianDistance, tempDist]
        C = np.argmin(EuclidianDistance, axis=1) + 1
        
        #UsedDistance = np.array([]).reshape(m,0)
        #for j in range(cluster_number):
        #    tempDist = []
        #    for _p in X :
        #        tempDist.append(distance_used(_p, centroid[:, j]))
        #    tempDist = np.array(tempDist)
        #    UsedDistance = np.c_[UsedDistance, tempDist]
        #C = np.argmin(UsedDistance, axis=1) + 1

        for k in range(cluster_number):
            Y[k+1] = np.array([]).reshape(n,0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]],X[i]]
        for k in range(cluster_number):
            Y[k+1] = Y[k+1].T
        for k in range(cluster_number):
            centroid[:,k] = np.mean(Y[k+1], axis=0)
            
    res_centroid = []
    for i in range(cluster_number):
        ci = [_cent[i] for _cent in centroid]
        res_centroid.append(ci)
        
    return Y, res_centroid

def K_Mean_DTW(X, n_iters, cluster_number):
    ## K_Mean for hard - clustering algorithm

    distance_used = distance_DTW
    m = X.shape[0] 
    n = X.shape[1]
    centroid = np.array([]).reshape(n, 0)
    for i in range(cluster_number):
        centroid = np.c_[centroid, X[rd.randint(0, m-1)]]
    Y = {}
    
    for i in tqdm(range(n_iters)):
        UsedDistance = np.array([]).reshape(m,0)
        for j in range(cluster_number):
            tempDist = []
            for _p in X :
                tempDist.append(distance_used(_p, centroid[:, j]))
            tempDist = np.array(tempDist)
            UsedDistance = np.c_[UsedDistance, tempDist]
        C = np.argmin(UsedDistance, axis=1) + 1

        for k in range(cluster_number):
            Y[k+1] = np.array([]).reshape(n,0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]],X[i]]
        for k in range(cluster_number):
            Y[k+1] = Y[k+1].T
        for k in range(cluster_number):
            centroid[:,k] = np.mean(Y[k+1], axis=0)
            
    res_centroid = []
    for i in range(cluster_number):
        ci = [_cent[i] for _cent in centroid]
        res_centroid.append(ci)
        
    return Y, res_centroid

from sklearn.cluster import DBSCAN
from sklearn import metrics

def _DBSCAN(X, eps, min_samples):
    ## DBSCAN for free - clustering algorithm
    
    db_default = DBSCAN(eps , min_samples).fit(X)
    labels_1 = db_default.labels_
    
    return labels_1



def set2List(NumpyArray):
    list = []
    for item in NumpyArray:
        list.append(item.tolist())
    return list


def _DBSCAN2(Dataset, Epsilon, MinumumPoints, DistanceMethod = 'euclidean'):
    import scipy
#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
    m, n = Dataset.shape
    Visited = np.zeros(m,'int')
    Type = np.zeros(m)
#   -1 noise, outlier
#    0 border
#    1 core
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=np.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset, DistanceMethod))
    for i in range(m):
        if Visited[i]==0:
            Visited[i]=1
            PointNeighbors=np.where(DistanceMatrix[i]<Epsilon)[0]
            if len(PointNeighbors)<MinumumPoints:
                Type[i]=-1
            else:
                for k in range(len(Cluster)):
                    Cluster.pop()
                Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
                
                
                PointNeighbors=set2List(PointNeighbors)    
                ExpandClsuter(Dataset[i], PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  )
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                 
                    
    return PointClusterNumber, ClusterList


def ExpandClsuter(PointToExapnd, PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  ):
    Neighbors=[]

    for i in PointNeighbors:
        if Visited[i]==0:
            Visited[i]=1
            Neighbors=np.where(DistanceMatrix[i]<Epsilon)[0]
            if len(Neighbors)>=MinumumPoints:
#                Neighbors merge with PointNeighbors
                for j in Neighbors:
                    try:
                        PointNeighbors.index(j)
                    except ValueError:
                        PointNeighbors.append(j)
                    
        if PointClusterNumber[i]==0:
            Cluster.append(i)
            PointClusterNumber[i]=PointClusterNumberIndex
    return