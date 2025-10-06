import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# helper function

def calculate_mean(X):

    """
    Return mean of given X
    """

    return np.mean(X, axis=0)

def cal_distance(x1, x2):

    """
    Calculate euclidean distance between two points
    """

    return np.sqrt(np.sum((x1-x2)**2))

def find_furthest_point(X):

    """
    Find first farthest point from centroid (P) and second farthest point (Q) from the first point (P)
    """

    centroid = calculate_mean(X)
    distances_1 = [cal_distance(x, centroid) for x in X]
    farthest_idx_1 = np.argmax(distances_1)
    farthest_value_1 = X[farthest_idx_1]
    distances_2 = [cal_distance(x, farthest_value_1) for x in X]
    farthest_idx_2 = np.argmax(distances_2)
    return farthest_idx_1, farthest_idx_2

def find_closest_point(reference_value, X, k=5):

    """
    Find k-1 points closest to reference point and return those points include reference point
    """

    distances = []
    for i, x in enumerate(X):
        dist = cal_distance(x, reference_value)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    closest_idx = [idx for idx, _ in distances[:k]]
    return closest_idx

def check_k_anonym(df, k):

    """
    
    params: 
    df: dataset after clustering
    k: minimum anonymity
    
    """

    min_anonym = min(df['cluster_id'].value_counts())
    if min_anonym < k:
        return False
    else:
        return min_anonym, True

def cal_info_loss(origin_df, anonymized_df, qasi_col:List=['Age']):
    total_sse = 0

    for col in qasi_col:
        original_val = origin_df[col].values
        anonymized_val = anonymized_df[col].values
        sse = np.sum((original_val - anonymized_val) ** 2)
        total_sse += sse
    
    return total_sse

def generalize_zip(zip, level=1):
    
    prefix = str(zip)[:-level]
    stars = '*' * level
    return prefix + stars

def anonymized_dataset(df, k:int=5, qasi_col:List=['Age']):

    """
    Return anonymized dataset and cluster assignment
    
    Algorithm adapted from Mortazavi et. al. (2014, 10.1016/j.knosys.2014.05.011)
    
    while 2k points or more
        find centroid C
        find furthest point P from C
        find furthest point Q from P
        cluster = P & (k-1) closest to P
        cluster = Q & (k-1) closest to Q
        remaining point = all except P-cluster and Q-cluster

    if less than 2k-1 points and more than k points
        form final cluster
    
    if less than k points:
        add to final cluster
        
    """
    label_encoder = LabelEncoder()

    anonymized_df = df.copy()
    anonymized_df['Sex'] = label_encoder.fit_transform(anonymized_df['Sex'])
    numeric_QASI = anonymized_df[qasi_col].values
    remaining_indices = list(range(len(anonymized_df)))
    cluster_id = 1
    cluster_assignments = np.zeros(len(anonymized_df), dtype=int)


    while len(remaining_indices) >= 2 * k:
        remaining_data = numeric_QASI[remaining_indices]

        farthest_idx_1, farthest_idx_2 = find_furthest_point(remaining_data)

        farthest_value_1 = remaining_data[farthest_idx_1]
        farthest_value_2 = remaining_data[farthest_idx_2]

        closest_idx_1 = find_closest_point(farthest_value_1, remaining_data, k)
        cluster_1_origin_indices = [remaining_indices[idx] for idx in closest_idx_1]
        for idx in cluster_1_origin_indices:
            cluster_assignments[idx] = cluster_id

        cluster_id += 1
        closest_idx_2 = find_closest_point(farthest_value_2, remaining_data, k)    
        cluster_2_origin_indices = [remaining_indices[idx] for idx in closest_idx_2]
        for idx in cluster_2_origin_indices:
            cluster_assignments[idx] = cluster_id

        cluster_id += 1
        all_clustered_idx = set(closest_idx_1 + closest_idx_2)
        remaining_indices = [remaining_indices[i] for i in range(len(remaining_indices)) if i not in all_clustered_idx]


    if k < len(remaining_indices) < 2 * k - 1:
        for idx in remaining_indices:
            cluster_assignments[idx] = cluster_id
    else:
        last_cluster_id = cluster_id - 1
        for idx in remaining_indices:
            cluster_assignments[idx] = last_cluster_id

    anonymized_df['cluster_id'] = cluster_assignments

    for cluster_id_val in np.unique(cluster_assignments):
        cluster_mask = cluster_assignments == cluster_id_val
        cluster_data = anonymized_df.loc[cluster_mask, qasi_col]
        cluster_sex = anonymized_df.loc[cluster_mask, 'Sex']
        unique_cluster_sex = np.unique(cluster_sex, return_counts=True)[0]
        unique_cluster_sex_count = np.unique(cluster_sex, return_counts=True)[1]
        if len(unique_cluster_sex) == 2:
            num_f, num_m = unique_cluster_sex_count
        else:
            if unique_cluster_sex[0] == 0:
                num_f = unique_cluster_sex_count[0]
                num_m = 0
            else:
                num_f = 0
                num_m = unique_cluster_sex_count[0]

        cluster_sex_mask = num_f / (num_f + num_m)
        cluster_means = round(cluster_data.mean(),0)
        for col in qasi_col:
            anonymized_df.loc[cluster_mask, col] = cluster_means[col]
        anonymized_df.loc[cluster_mask, 'Sex'] = None
        anonymized_df.loc[cluster_mask, 'Sex'] = cluster_sex_mask
    anonymized_df['ZIP'] = anonymized_df['ZIP'].apply(generalize_zip)
    
    num_clusters = len(np.unique(cluster_assignments))
    min_cluster_size, is_k_anonym = check_k_anonym(anonymized_df, k)

    print(f'Number of clusters formed: {num_clusters}')
    print(f'Minimum cluster size: {min_cluster_size}')
    print(f'{k}-anonymity satisfied: {is_k_anonym}')

    return anonymized_df